#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "open_spiel/algorithms/external_sampling_mccfr.h"
#include "open_spiel/examples/hu_limit_bot_game.h"
#include "open_spiel/spiel.h"
#include "open_spiel/games/universal_poker/universal_poker.h"

namespace {

uint64_t Checksum(const std::string& bytes) {
  uint64_t hash = 14695981039346656037ull;
  for (unsigned char c : bytes) hash = (hash ^ c) * 1099511628211ull;
  return hash;
}

void WriteU16(std::ostream& out, uint16_t value) {
  const std::array<char, 2> bytes{static_cast<char>(value),
                                  static_cast<char>(value >> 8)};
  out.write(bytes.data(), bytes.size());
}

void WriteU32(std::ostream& out, uint32_t value) {
  const std::array<char, 4> bytes{static_cast<char>(value),
                                  static_cast<char>(value >> 8),
                                  static_cast<char>(value >> 16),
                                  static_cast<char>(value >> 24)};
  out.write(bytes.data(), bytes.size());
}

void WriteDouble(std::ostream& out, double value) {
  static_assert(sizeof(double) == sizeof(uint64_t));
  uint64_t bits;
  std::memcpy(&bits, &value, sizeof(bits));
  for (int shift = 0; shift < 64; shift += 8) {
    out.put(static_cast<char>(bits >> shift));
  }
}

void Export(const std::filesystem::path& checkpoint,
            const std::filesystem::path& destination) {
  std::ifstream input(checkpoint, std::ios::binary);
  if (!input) throw std::runtime_error("cannot open checkpoint");
  std::string line;
  std::getline(input, line);
  std::istringstream header(line);
  std::string magic;
  uint64_t iterations = 0, checksum = 0;
  double elapsed = 0;
  size_t size = 0;
  if (!(header >> magic >> iterations >> elapsed >> size >> checksum) ||
      magic != "HULHE1" || !std::isfinite(elapsed) ||
      size > (size_t{2} << 30)) {
    throw std::runtime_error("invalid checkpoint header");
  }
  std::string serialized(size, '\0');
  if (!input.read(serialized.data(), size) || Checksum(serialized) != checksum) {
    throw std::runtime_error("checkpoint checksum mismatch");
  }
  const auto game = open_spiel::LoadGame(open_spiel::hu_limit_bot::kGameString);
  const std::string game_block = "[Game]\n" + game->Serialize() + "\n";
  const size_t game_offset = serialized.find("[Game]\n");
  if (game_offset == std::string::npos ||
      serialized.compare(game_offset, game_block.size(), game_block) != 0) {
    throw std::runtime_error("unexpected checkpoint game");
  }
  auto solver = open_spiel::algorithms::DeserializeExternalSamplingMCCFRSolver(
      serialized);
  const auto& table = solver->InfoStateValuesTable();
  if (table.empty() || table.size() > std::numeric_limits<uint32_t>::max()) {
    throw std::runtime_error("invalid policy table size");
  }
  std::vector<std::string> keys;
  keys.reserve(table.size());
  for (const auto& [key, _] : table) keys.push_back(key);
  std::sort(keys.begin(), keys.end());

  std::ofstream output(destination, std::ios::binary | std::ios::trunc);
  if (!output) throw std::runtime_error("cannot create model");
  output.write("HULHEP1\0", 8);
  WriteU32(output, keys.size());
  for (const auto& key : keys) {
    const auto& values = table.at(key);
    if (key.size() > std::numeric_limits<uint16_t>::max()) {
      throw std::runtime_error("oversized key");
    }
    WriteU16(output, key.size());
    output.write(key.data(), key.size());
    std::array<double, 3> probabilities{};
    if (values.legal_actions.empty() ||
        values.legal_actions.size() != values.cumulative_policy.size()) {
      throw std::runtime_error("invalid action/weight count");
    }
    double sum = 0;
    for (double weight : values.cumulative_policy) {
      if (!std::isfinite(weight) || weight < 0)
        throw std::runtime_error("invalid policy weight");
      sum += weight;
    }
    if (!std::isfinite(sum)) throw std::runtime_error("invalid policy sum");
    int seen = 0;
    for (size_t i = 0; i < values.legal_actions.size(); ++i) {
      const int action = values.legal_actions[i];
      if (action < 0 || action > 2 || (seen & (1 << action)))
        throw std::runtime_error("unexpected or repeated action");
      seen |= 1 << action;
      probabilities[action] = sum > 0
          ? values.cumulative_policy[i] / sum
          : 1.0 / values.legal_actions.size();
    }
    for (double probability : probabilities) WriteDouble(output, probability);
  }
  output.flush();
  if (!output) throw std::runtime_error("model write failed");
  std::cerr << "exported " << keys.size() << " policy keys from "
            << iterations << " iterations / " << elapsed << " seconds to "
            << destination << '\n';
}

void Goldens(const std::filesystem::path& destination, int games) {
  auto game = open_spiel::LoadGame(open_spiel::hu_limit_bot::kGameString);
  std::mt19937 rng(20260918);
  std::ofstream out(destination);
  if (!out) throw std::runtime_error("cannot create golden file");
  constexpr std::array<int, 4> counts{0, 3, 4, 5};
  for (int g = 0; g < games; ++g) {
    auto state = game->NewInitialState();
    while (!state->IsTerminal()) {
      if (!state->IsChanceNode()) {
        const auto& wrapper = static_cast<const open_spiel::hu_limit_bot::AbstractState&>(*state);
        const auto& poker = static_cast<const open_spiel::universal_poker::UniversalPokerState&>(
            wrapper.GetWrappedState());
        const auto& acpc = poker.acpc_state();
        const int round = acpc.GetRound(), player = state->CurrentPlayer();
        const std::string key = state->InformationStateString(player);
        const size_t b = key.rfind("|b"), m = key.rfind("|m");
        if (b == std::string::npos || m == std::string::npos)
          throw std::runtime_error("invalid abstraction key");
        out << round << ' ' << int(acpc.hole_cards(player, 0)) << ' '
            << int(acpc.hole_cards(player, 1));
        for (int i = 0; i < counts.at(round); ++i) out << ' ' << int(acpc.board_cards(i));
        out << ' ' << key.substr(b + 2, m - b - 2) << '\n';
      }
      const auto legal = state->LegalActions();
      if (legal.empty()) throw std::runtime_error("no legal action");
      std::uniform_int_distribution<size_t> dist(0, legal.size() - 1);
      state->ApplyAction(legal[dist(rng)]);
    }
  }
  if (!out) throw std::runtime_error("golden write failed");
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 3 && argc != 4) {
    std::cerr << "usage: arena_model_export CHECKPOINT MODEL | --golden OUT GAMES\n";
    return 2;
  }
  try {
    if (argc == 4 && std::string(argv[1]) == "--golden")
      Goldens(argv[2], std::stoi(argv[3]));
    else if (argc == 3)
      Export(argv[1], argv[2]);
    else throw std::runtime_error("invalid arguments");
  } catch (const std::exception& error) {
    std::cerr << "arena_model_export: " << error.what() << '\n';
    return 1;
  }
}
