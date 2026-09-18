#include "open_spiel/examples/hu_limit_bot_game.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <memory>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "open_spiel/abseil-cpp/absl/strings/str_cat.h"
#include "open_spiel/games/universal_poker/logic/card_set.h"
#include "open_spiel/games/universal_poker/universal_poker.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"

namespace open_spiel::hu_limit_bot {
namespace {

constexpr char kBaseGame[] =
    "universal_poker(betting=limit,numPlayers=2,numRounds=4,blind=2 1,"
    "firstPlayer=2 1 1 1,raiseSize=2 2 4 4,maxRaises=3 4 4 4,"
    "numSuits=4,numRanks=13,numHoleCards=2,numBoardCards=0 3 1 1,"
    "bettingAbstraction=fcpa,calcOddsNumSims=0)";

const GameType kGameType{
    "hu_limit_holdem", "Heads-up fixed-limit Hold'em card abstraction",
    GameType::Dynamics::kSequential,
    GameType::ChanceMode::kExplicitStochastic,
    GameType::Information::kImperfectInformation,
    GameType::Utility::kZeroSum,
    GameType::RewardModel::kTerminal,
    2, 2,
    /*provides_information_state_string=*/true,
    /*provides_information_state_tensor=*/false,
    /*provides_observation_string=*/true,
    /*provides_observation_tensor=*/true,
    {{"equity_bins", GameParameter(32)},
     {"equity_samples", GameParameter(128)}}};

std::shared_ptr<const Game> Factory(const GameParameters& params) {
  auto base_game = LoadGame(kBaseGame);
  return std::make_shared<AbstractGame>(std::move(base_game), kGameType,
                                        params);
}

REGISTER_SPIEL_GAME(kGameType, Factory);

const universal_poker::UniversalPokerState& PokerState(const State& state) {
  return static_cast<const universal_poker::UniversalPokerState&>(
      static_cast<const AbstractState&>(state).GetWrappedState());
}

uint32_t VisibleSeed(const std::array<int, 2>& hole,
                     const std::vector<int>& board, int round) {
  // FNV-1a over visible cards, independent of the game's undealt cards.
  uint32_t hash = 2166136261u;
  for (int card : hole) hash = (hash ^ static_cast<uint32_t>(card + 1)) * 16777619u;
  for (int card : board) hash = (hash ^ static_cast<uint32_t>(card + 1)) * 16777619u;
  return (hash ^ static_cast<uint32_t>(round + 1)) * 16777619u;
}

int PreflopBucket(const std::array<int, 2>& hole) {
  const int rank_a = hole[0] / 4;
  const int rank_b = hole[1] / 4;
  if (rank_a == rank_b) return rank_a;
  const int high = std::max(rank_a, rank_b);
  const int low = std::min(rank_a, rank_b);
  const int pair_index = high * (high - 1) / 2 + low;
  return (hole[0] % 4 == hole[1] % 4 ? 13 : 91) + pair_index;
}

double Equity(const std::array<int, 2>& hole, const std::vector<int>& board,
              int round, int samples) {
  std::array<bool, 52> used{};
  used[hole[0]] = true;
  used[hole[1]] = true;
  for (int card : board) used[card] = true;

  std::vector<int> remaining;
  remaining.reserve(50 - board.size());
  for (int card = 0; card < 52; ++card) {
    if (!used[card]) remaining.push_back(card);
  }

  std::mt19937 rng(VisibleSeed(hole, board, round));
  double wins = 0.0;
  const int draw_count = 2 + 5 - board.size();
  for (int sample = 0; sample < samples; ++sample) {
    // Shuffle only the cards required for the opponent hand and future board.
    for (int i = 0; i < draw_count; ++i) {
      std::uniform_int_distribution<int> dist(i, remaining.size() - 1);
      std::swap(remaining[i], remaining[dist(rng)]);
    }
    universal_poker::logic::CardSet mine;
    universal_poker::logic::CardSet theirs;
    mine.AddCard(hole[0]);
    mine.AddCard(hole[1]);
    theirs.AddCard(remaining[0]);
    theirs.AddCard(remaining[1]);
    for (int card : board) {
      mine.AddCard(card);
      theirs.AddCard(card);
    }
    for (int i = 2; i < draw_count; ++i) {
      mine.AddCard(remaining[i]);
      theirs.AddCard(remaining[i]);
    }
    const int my_rank = mine.RankCards();
    const int their_rank = theirs.RankCards();
    wins += (my_rank > their_rank ? 1.0 : my_rank == their_rank ? 0.5 : 0.0);
  }
  return wins / samples;
}

}  // namespace

AbstractState::AbstractState(std::shared_ptr<const Game> game,
                             std::unique_ptr<State> state, int equity_bins,
                             int equity_samples)
    : WrappedState(std::move(game), std::move(state)),
      equity_bins_(equity_bins), equity_samples_(equity_samples) {}

int AbstractState::CardBucket(Player player) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, 2);
  if (cached_bucket_[player] >= 0) return cached_bucket_[player];
  const auto& acpc = PokerState(*this).acpc_state();
  const int round = acpc.GetRound();
  std::array<int, 2> hole{acpc.hole_cards(player, 0),
                          acpc.hole_cards(player, 1)};
  std::sort(hole.begin(), hole.end());
  if (round == 0) return cached_bucket_[player] = PreflopBucket(hole);
  const double equity = EstimatedEquity(player);
  return cached_bucket_[player] =
             std::min(equity_bins_ - 1, static_cast<int>(equity * equity_bins_));
}

double AbstractState::EstimatedEquity(Player player) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, 2);
  SPIEL_CHECK_FALSE(IsChanceNode());
  if (cached_equity_[player] >= 0) return cached_equity_[player];
  const auto& acpc = PokerState(*this).acpc_state();
  const int round = acpc.GetRound();
  std::array<int, 2> hole{acpc.hole_cards(player, 0),
                          acpc.hole_cards(player, 1)};
  std::sort(hole.begin(), hole.end());
  constexpr std::array<int, 4> kBoardCounts{0, 3, 4, 5};
  std::vector<int> board;
  for (int i = 0; i < kBoardCounts.at(round); ++i) {
    board.push_back(acpc.board_cards(i));
  }
  std::sort(board.begin(), board.end());
  return cached_equity_[player] = Equity(hole, board, round, equity_samples_);
}

std::string AbstractState::InformationStateString(Player player) const {
  SPIEL_CHECK_FALSE(IsChanceNode());
  SPIEL_CHECK_FALSE(IsTerminal());
  const auto& acpc = PokerState(*this).acpc_state();
  const int round = acpc.GetRound();
  std::string key = absl::StrCat("v1|p", player, "|r", round, "|s");
  for (int r = 0; r <= round; ++r) {
    absl::StrAppend(&key, r == 0 ? "" : "/", acpc.BettingSequence(r));
  }
  int legal_mask = 0;
  for (Action action : LegalActions()) {
    SPIEL_CHECK_GE(action, 0);
    SPIEL_CHECK_LE(action, 2);
    legal_mask |= 1 << action;
  }
  absl::StrAppend(&key, "|b", CardBucket(player), "|m", legal_mask);
  return key;
}

void AbstractState::InformationStateTensor(Player,
                                           absl::Span<float>) const {
  SpielFatalError("hu_limit_holdem provides only abstract information strings");
}

std::unique_ptr<State> AbstractState::Clone() const {
  return std::make_unique<AbstractState>(*this);
}

void AbstractState::DoApplyAction(Action action) {
  const bool chance = IsChanceNode();
  WrappedState::DoApplyAction(action);
  if (chance) {
    cached_bucket_.fill(-1);
    cached_equity_.fill(-1.0);
  }
}

AbstractGame::AbstractGame(std::shared_ptr<const Game> game, GameType game_type,
                           GameParameters params)
    : WrappedGame(std::move(game), std::move(game_type), std::move(params)),
      equity_bins_(ParameterValue<int>("equity_bins")),
      equity_samples_(ParameterValue<int>("equity_samples")) {
  SPIEL_CHECK_GE(equity_bins_, 2);
  SPIEL_CHECK_LE(equity_bins_, 128);
  SPIEL_CHECK_GE(equity_samples_, 1);
  SPIEL_CHECK_LE(equity_samples_, 4096);
}

std::unique_ptr<State> AbstractGame::NewInitialState() const {
  return std::make_unique<AbstractState>(shared_from_this(),
                                          game_->NewInitialState(), equity_bins_,
                                          equity_samples_);
}

}  // namespace open_spiel::hu_limit_bot
