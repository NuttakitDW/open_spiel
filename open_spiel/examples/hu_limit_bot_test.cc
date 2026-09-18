#include <algorithm>
#include <cmath>
#include <memory>
#include <string>
#include <vector>

#include "open_spiel/algorithms/external_sampling_mccfr.h"
#include "open_spiel/examples/hu_limit_bot_game.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"

namespace os = open_spiel;

namespace {

void Deal(os::State& state, const std::vector<int>& cards) {
  for (int card : cards) {
    SPIEL_CHECK_TRUE(state.IsChanceNode());
    const auto legal = state.LegalActions();
    SPIEL_CHECK_TRUE(std::find(legal.begin(), legal.end(), card) != legal.end());
    state.ApplyAction(card);
  }
}

std::unique_ptr<os::State> Preflop(const os::Game& game,
                                  const std::vector<int>& hole) {
  auto state = game.NewInitialState();
  Deal(*state, hole);
  SPIEL_CHECK_FALSE(state->IsChanceNode());
  return state;
}

void TestRulesAndInformation() {
  auto game = os::LoadGame(os::hu_limit_bot::kGameString);
  SPIEL_CHECK_EQ(game->NumPlayers(), 2);
  SPIEL_CHECK_EQ(game->MaxUtility(), 48);
  SPIEL_CHECK_FALSE(game->GetType().provides_information_state_tensor);

  auto first = Preflop(*game, {0, 1, 2, 3});
  auto second = Preflop(*game, {4, 5, 2, 3});
  SPIEL_CHECK_EQ(first->CurrentPlayer(), 1);  // Button/small blind first.
  SPIEL_CHECK_EQ(first->InformationStateString(1),
                 second->InformationStateString(1));
  const auto& first_abstract =
      static_cast<const os::hu_limit_bot::AbstractState&>(*first);
  const auto& second_abstract =
      static_cast<const os::hu_limit_bot::AbstractState&>(*second);
  SPIEL_CHECK_EQ(first_abstract.EstimatedEquity(1),
                 second_abstract.EstimatedEquity(1));
  SPIEL_CHECK_NE(first->InformationStateString(0),
                 second->InformationStateString(0));
  SPIEL_CHECK_EQ(first->LegalActions(), second->LegalActions());

  first->ApplyAction(1);   // Button calls big blind.
  second->ApplyAction(1);
  SPIEL_CHECK_EQ(first->CurrentPlayer(), 0);
  first->ApplyAction(1);   // Big blind checks.
  second->ApplyAction(1);
  Deal(*first, {10, 11, 12});
  Deal(*second, {10, 11, 12});
  SPIEL_CHECK_EQ(first->CurrentPlayer(), 0);  // Big blind first postflop.
  SPIEL_CHECK_EQ(first->InformationStateString(1),
                 second->InformationStateString(1));

  auto cloned = first->Clone();
  SPIEL_CHECK_EQ(cloned->InformationStateString(1),
                 first->InformationStateString(1));
  first->ApplyAction(1);
  SPIEL_CHECK_NE(cloned->InformationStateString(1),
                 first->InformationStateString(1));
}

void TestRaiseCapsAndTerminalReturns() {
  auto game = os::LoadGame(os::hu_limit_bot::kGameString);
  auto state = Preflop(*game, {0, 1, 2, 3});
  const std::vector<int> expected_raises{3, 4, 4, 4};
  int board_card = 10;
  for (int round = 0; round < 4; ++round) {
    int raises = 0;
    while (true) {
      const auto legal = state->LegalActions();
      if (std::find(legal.begin(), legal.end(), 2) == legal.end()) break;
      state->ApplyAction(2);
      ++raises;
    }
    SPIEL_CHECK_EQ(raises, expected_raises.at(round));
    state->ApplyAction(1);  // Last player calls the capped raise.
    if (round == 3) break;
    const int cards = round == 0 ? 3 : 1;
    for (int i = 0; i < cards; ++i) {
      SPIEL_CHECK_TRUE(state->IsChanceNode());
      state->ApplyAction(board_card++);
    }
  }
  SPIEL_CHECK_TRUE(state->IsTerminal());
  const auto returns = state->Returns();
  SPIEL_CHECK_EQ(returns.size(), 2);
  SPIEL_CHECK_EQ(returns[0] + returns[1], 0);
  SPIEL_CHECK_LE(std::abs(returns[0]), 48);
}

void TestSolverResume() {
  auto game = os::LoadGame(os::hu_limit_bot::kGameString);
  os::algorithms::ExternalSamplingMCCFRSolver uninterrupted(*game, 42);
  for (int i = 0; i < 10; ++i) uninterrupted.RunIteration();
  auto resumed = os::algorithms::DeserializeExternalSamplingMCCFRSolver(
      uninterrupted.Serialize());
  auto check_equal = [&]() {
    const auto& left = uninterrupted.InfoStateValuesTable();
    const auto& right = resumed->InfoStateValuesTable();
    SPIEL_CHECK_EQ(left.size(), right.size());
    for (const auto& [key, values] : left) {
      SPIEL_CHECK_TRUE(right.find(key) != right.end());
      SPIEL_CHECK_TRUE(values.Serialize(-1) == right.at(key).Serialize(-1));
    }
    const std::string left_solver = uninterrupted.Serialize();
    const std::string right_solver = resumed->Serialize();
    const std::string marker = "[SolverValuesTable]";
    SPIEL_CHECK_TRUE(left_solver.substr(0, left_solver.find(marker)) ==
                     right_solver.substr(0, right_solver.find(marker)));
  };
  check_equal();
  for (int i = 0; i < 10; ++i) {
    uninterrupted.RunIteration();
    resumed->RunIteration();
  }
  check_equal();
}

void TestRoyalFlushBucket() {
  auto game = os::LoadGame(os::hu_limit_bot::kGameString);
  auto state = Preflop(*game, {51, 47, 4, 8});  // As Ks vs 3c 4c.
  state->ApplyAction(1);
  state->ApplyAction(1);
  Deal(*state, {43, 39, 35});  // Qs Js Ts.
  state->ApplyAction(1);
  state->ApplyAction(1);
  Deal(*state, {0});  // 2c.
  state->ApplyAction(1);
  state->ApplyAction(1);
  Deal(*state, {5});  // 3d.
  SPIEL_CHECK_TRUE(state->InformationStateString(0).find("|b31|") !=
                   std::string::npos);
  SPIEL_CHECK_EQ(static_cast<const os::hu_limit_bot::AbstractState&>(*state)
                     .EstimatedEquity(0), 1.0);
}

}  // namespace

int main() {
  TestRulesAndInformation();
  TestRaiseCapsAndTerminalReturns();
  TestSolverResume();
  TestRoyalFlushBucket();
}
