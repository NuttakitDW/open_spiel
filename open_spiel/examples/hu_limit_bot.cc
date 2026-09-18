#include <sys/resource.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "open_spiel/algorithms/external_sampling_mccfr.h"
#include "open_spiel/examples/hu_limit_bot_game.h"
#include "open_spiel/game_parameters.h"
#include "open_spiel/games/universal_poker/logic/card_set.h"
#include "open_spiel/games/universal_poker/universal_poker.h"
#include "open_spiel/policy.h"
#include "open_spiel/spiel.h"

namespace fs = std::filesystem;
namespace os = open_spiel;
using Solver = os::algorithms::ExternalSamplingMCCFRSolver;
using Clock = std::chrono::steady_clock;

namespace {

struct Options {
  std::string mode;
  fs::path checkpoint = "checkpoints/hu_limit_holdem/current.chk";
  fs::path status = "checkpoints/hu_limit_holdem/status.txt";
  double seconds = 43200;
  int64_t deadline_epoch = 0;
  int checkpoint_interval = 600;
  int seed = 20260917;
  bool seed_explicit = false;
  int hands = 1000;
  int human_seat = 1;
  bool final_eval = true;
  size_t max_nodes = 500000;
  size_t max_rss_bytes = size_t{4} << 30;
};

Options ParseOptions(int argc, char** argv) {
  Options options;
  if (argc >= 2) options.mode = argv[1];
  for (int i = 2; i < argc; ++i) {
    std::string argument = argv[i];
    const size_t equals = argument.find('=');
    if (argument.rfind("--", 0) != 0 || equals == std::string::npos) {
      throw std::invalid_argument("expected --name=value: " + argument);
    }
    const std::string name = argument.substr(2, equals - 2);
    const std::string value = argument.substr(equals + 1);
    auto integer = [&]() {
      size_t consumed = 0;
      const int64_t parsed = std::stoll(value, &consumed);
      if (consumed != value.size()) throw std::invalid_argument(argument);
      return parsed;
    };
    auto small_integer = [&]() {
      const int64_t parsed = integer();
      if (parsed < std::numeric_limits<int>::min() ||
          parsed > std::numeric_limits<int>::max()) {
        throw std::invalid_argument(argument);
      }
      return static_cast<int>(parsed);
    };
    auto positive_size = [&]() {
      const int64_t parsed = integer();
      if (parsed <= 0) throw std::invalid_argument(argument);
      return static_cast<size_t>(parsed);
    };
    if (name == "checkpoint") options.checkpoint = value;
    else if (name == "status") options.status = value;
    else if (name == "seconds") {
      size_t consumed = 0;
      options.seconds = std::stod(value, &consumed);
      if (consumed != value.size()) throw std::invalid_argument(argument);
    }
    else if (name == "deadline_epoch") options.deadline_epoch = integer();
    else if (name == "checkpoint_interval") options.checkpoint_interval = small_integer();
    else if (name == "seed") {
      options.seed = small_integer();
      options.seed_explicit = true;
    }
    else if (name == "hands") options.hands = small_integer();
    else if (name == "human_seat") options.human_seat = small_integer();
    else if (name == "final_eval") {
      const int parsed = small_integer();
      if (parsed != 0 && parsed != 1) throw std::invalid_argument(argument);
      options.final_eval = parsed == 1;
    }
    else if (name == "max_nodes") options.max_nodes = positive_size();
    else if (name == "max_rss_bytes") options.max_rss_bytes = positive_size();
    else throw std::invalid_argument("unknown option: " + name);
  }
  if (!std::isfinite(options.seconds) || options.seconds < 0 ||
      options.deadline_epoch < 0 || options.checkpoint_interval < 1 ||
      options.hands < 1 || options.human_seat < 0 || options.human_seat > 1 ||
      options.max_nodes == 0 || options.max_rss_bytes == 0 ||
      options.seed > std::numeric_limits<int>::max() - 3000) {
    throw std::invalid_argument("invalid numeric option");
  }
  return options;
}

fs::path ParentOrDot(const fs::path& path) {
  return path.has_parent_path() ? path.parent_path() : fs::path(".");
}

std::string UtcNow() {
  const auto now = std::chrono::system_clock::now();
  const std::time_t seconds = std::chrono::system_clock::to_time_t(now);
  std::tm tm{};
  gmtime_r(&seconds, &tm);
  std::ostringstream result;
  result << std::put_time(&tm, "%Y-%m-%dT%H:%M:%SZ");
  return result.str();
}

uint64_t Checksum(const std::string& data) {
  uint64_t hash = 14695981039346656037ull;
  for (unsigned char character : data) {
    hash = (hash ^ character) * 1099511628211ull;
  }
  return hash;
}

size_t PeakRssBytes() {
  rusage usage{};
  if (getrusage(RUSAGE_SELF, &usage) != 0) return 0;
#ifdef __APPLE__
  return usage.ru_maxrss;
#else
  return usage.ru_maxrss * 1024;
#endif
}

struct TrainingState {
  uint64_t iterations = 0;
  double elapsed_seconds = 0;
  std::unique_ptr<Solver> solver;
};

void AtomicText(const fs::path& path, const std::string& data) {
  fs::create_directories(ParentOrDot(path));
  const fs::path temporary = path.string() + ".tmp";
  {
    std::ofstream output(temporary, std::ios::binary | std::ios::trunc);
    if (!output.write(data.data(), data.size()) || !output.flush()) {
      throw std::runtime_error("could not write " + temporary.string());
    }
  }
  fs::rename(temporary, path);
}

void SaveCheckpoint(const fs::path& path, const TrainingState& state) {
  fs::create_directories(ParentOrDot(path));
  const std::string solver = state.solver->Serialize();
  const uintmax_t available = fs::space(ParentOrDot(path)).available;
  constexpr uintmax_t kReserve = uintmax_t{3} << 30;
  if (available < kReserve + solver.size()) {
    throw std::runtime_error("less than 3 GiB free after checkpoint write");
  }
  std::ostringstream header;
  header << "HULHE1 " << state.iterations << ' ' << std::setprecision(17)
         << state.elapsed_seconds << ' ' << solver.size() << ' '
         << Checksum(solver) << '\n';
  const fs::path temporary = path.string() + ".tmp";
  {
    std::ofstream output(temporary, std::ios::binary | std::ios::trunc);
    const std::string header_text = header.str();
    if (!output.write(header_text.data(), header_text.size()) ||
        !output.write(solver.data(), solver.size()) || !output.flush()) {
      throw std::runtime_error("could not write checkpoint");
    }
  }
  const fs::path previous = path.string() + ".previous";
  if (fs::exists(path)) {
    std::error_code ignored;
    fs::remove(previous, ignored);
    fs::rename(path, previous);
  }
  fs::rename(temporary, path);
}

TrainingState ReadCheckpoint(const fs::path& path, const os::Game& game) {
  std::ifstream input(path, std::ios::binary);
  if (!input) throw std::runtime_error("cannot open " + path.string());
  std::string line;
  std::getline(input, line);
  std::istringstream header(line);
  std::string magic;
  TrainingState state;
  size_t size = 0;
  uint64_t checksum = 0;
  if (!(header >> magic >> state.iterations >> state.elapsed_seconds >> size >>
        checksum) || magic != "HULHE1" || size > (size_t{2} << 30) ||
      !std::isfinite(state.elapsed_seconds) || state.elapsed_seconds < 0) {
    throw std::runtime_error("invalid checkpoint header");
  }
  std::string solver(size, '\0');
  if (!input.read(solver.data(), size) || Checksum(solver) != checksum) {
    throw std::runtime_error("incomplete or corrupt checkpoint");
  }
  const std::string game_block = "[Game]\n" + game.Serialize() + "\n";
  const size_t game_start = solver.find("[Game]\n");
  if (game_start == std::string::npos ||
      solver.compare(game_start, game_block.size(), game_block) != 0) {
    throw std::runtime_error("checkpoint game or abstraction differs");
  }
  state.solver = os::algorithms::DeserializeExternalSamplingMCCFRSolver(solver);
  return state;
}

TrainingState LoadOrCreate(const Options& options, const os::Game& game) {
  if (fs::exists(options.checkpoint)) {
    try {
      return ReadCheckpoint(options.checkpoint, game);
    } catch (const std::exception& error) {
      std::cerr << "current checkpoint failed: " << error.what() << '\n';
      const fs::path previous = options.checkpoint.string() + ".previous";
      if (!fs::exists(previous)) throw;
      TrainingState recovered = ReadCheckpoint(previous, game);
      fs::remove(options.checkpoint);
      return recovered;
    }
  }
  const fs::path previous = options.checkpoint.string() + ".previous";
  if (fs::exists(previous)) return ReadCheckpoint(previous, game);
  TrainingState state;
  state.solver = std::make_unique<Solver>(game, options.seed);
  return state;
}

void WriteStatus(const Options& options, const TrainingState& state,
                 const std::string& phase) {
  std::ostringstream status;
  status << "timestamp_utc=" << UtcNow() << '\n'
         << "phase=" << phase << '\n'
         << "iterations=" << state.iterations << '\n'
         << "training_seconds=" << std::fixed << std::setprecision(1)
         << state.elapsed_seconds << '\n'
         << "information_states=" << state.solver->InfoStateValuesTable().size()
         << '\n'
         << "peak_rss_bytes=" << PeakRssBytes() << '\n';
  AtomicText(options.status, status.str());
  std::cout << status.str() << std::flush;
}

os::Action RandomAction(const os::State& state, std::mt19937& rng) {
  const auto legal = state.LegalActions();
  std::uniform_int_distribution<size_t> distribution(0, legal.size() - 1);
  return legal[distribution(rng)];
}

os::Action PolicyAction(const os::State& state, const os::Policy& policy,
                        std::mt19937& rng) {
  const auto legal = state.LegalActions();
  const auto probabilities = policy.GetStatePolicy(state);
  std::vector<double> weights;
  weights.reserve(legal.size());
  for (os::Action action : legal) {
    double probability = 0;
    for (const auto& [candidate, weight] : probabilities) {
      if (candidate == action) probability = std::max(0.0, weight);
    }
    weights.push_back(probability);
  }
  if (std::all_of(weights.begin(), weights.end(),
                  [](double weight) { return weight == 0; })) {
    return RandomAction(state, rng);
  }
  std::discrete_distribution<size_t> distribution(weights.begin(),
                                                  weights.end());
  return legal[distribution(rng)];
}

void DealChance(os::State& state, std::mt19937& rng) {
  // A full-deck game has equal probability for each currently legal card.
  state.ApplyAction(RandomAction(state, rng));
}

os::Action HeuristicAction(const os::State& state) {
  const int player = state.CurrentPlayer();
  const auto& abstract_state =
      static_cast<const os::hu_limit_bot::AbstractState&>(state);
  const auto& poker = static_cast<const os::universal_poker::UniversalPokerState&>(
      abstract_state.GetWrappedState());
  const auto& acpc = poker.acpc_state();
  const double equity = abstract_state.EstimatedEquity(player);
  const double to_call = acpc.MaxSpend() - acpc.CurrentSpent(player);
  const double pot_odds = to_call / (acpc.TotalSpent() + to_call);
  const auto legal = state.LegalActions();
  if (std::find(legal.begin(), legal.end(), 2) != legal.end() &&
      equity >= std::max(0.68, pot_odds + 0.25)) {
    return 2;
  }
  if (to_call > 0 && equity + 0.05 < pot_odds &&
      std::find(legal.begin(), legal.end(), 0) != legal.end()) {
    return 0;
  }
  return 1;
}

struct Evaluation {
  int hands = 0;
  int seats[2] = {0, 0};
  uint64_t bot_decisions = 0;
  uint64_t unseen_decisions = 0;
  double sum = 0;
  double sum_squares = 0;
};

Evaluation Evaluate(const os::Game& game, const Solver& solver,
                    const std::string& opponent, int hands, int seed) {
  Evaluation result;
  auto policy = solver.AveragePolicy();
  auto& table = const_cast<Solver&>(solver).InfoStateValuesTable();
  std::mt19937 rng(seed);
  for (int hand = 0; hand < hands; ++hand) {
    const int bot_seat = hand % 2;
    ++result.seats[bot_seat];
    auto state = game.NewInitialState();
    while (!state->IsTerminal()) {
      if (state->IsChanceNode()) {
        DealChance(*state, rng);
      } else if (state->CurrentPlayer() == bot_seat) {
        ++result.bot_decisions;
        if (table.find(state->InformationStateString(bot_seat)) == table.end()) {
          ++result.unseen_decisions;
        }
        state->ApplyAction(PolicyAction(*state, *policy, rng));
      } else if (opponent == "call") {
        const auto legal = state->LegalActions();
        state->ApplyAction(std::find(legal.begin(), legal.end(), 1) != legal.end()
                               ? 1 : legal.front());
      } else if (opponent == "heuristic") {
        state->ApplyAction(HeuristicAction(*state));
      } else {
        state->ApplyAction(RandomAction(*state, rng));
      }
    }
    const double chips = state->Returns().at(bot_seat);
    result.sum += chips;
    result.sum_squares += chips * chips;
    ++result.hands;
  }
  return result;
}

void PrintEvaluation(const std::string& name, const Evaluation& result) {
  const double mean = result.sum / result.hands;
  const double variance = result.hands > 1
      ? std::max(0.0, (result.sum_squares - result.hands * mean * mean) /
                           (result.hands - 1)) : 0.0;
  const double bb100 = mean / 2.0 * 100.0;
  const double ci = 1.96 * std::sqrt(variance / result.hands) / 2.0 * 100.0;
  std::cout << name << ": hands=" << result.hands
            << " seats=" << result.seats[0] << '/' << result.seats[1]
            << " bb/100=" << std::fixed << std::setprecision(2) << bb100
            << " 95%CI=[" << bb100 - ci << ',' << bb100 + ci << ']'
            << " fallback=" << result.unseen_decisions << '/'
            << result.bot_decisions << '\n';
}

void EvaluateBoth(const os::Game& game, const Solver& solver,
                  const Options& options) {
  PrintEvaluation("random", Evaluate(game, solver, "random", options.hands,
                                     options.seed + 1000));
  PrintEvaluation("always_call", Evaluate(game, solver, "call", options.hands,
                                          options.seed + 2000));
  PrintEvaluation("visible_equity_heuristic",
                  Evaluate(game, solver, "heuristic", options.hands,
                           options.seed + 3000));
}

bool Train(const Options& options, const os::Game& game) {
  TrainingState state = LoadOrCreate(options, game);
  WriteStatus(options, state, "training");
  auto batch_start = Clock::now();
  auto last_checkpoint = batch_start;
  auto last_status = batch_start;
  std::string stopped = "complete";
  while (state.elapsed_seconds < options.seconds) {
    if (options.deadline_epoch > 0 &&
        std::time(nullptr) >= options.deadline_epoch) {
      stopped = "deadline";
      break;
    }
    state.solver->RunIteration();
    ++state.iterations;
    const auto now = Clock::now();
    state.elapsed_seconds += std::chrono::duration<double>(now - batch_start).count();
    batch_start = now;
    if (state.solver->InfoStateValuesTable().size() > options.max_nodes ||
        PeakRssBytes() > options.max_rss_bytes) {
      stopped = "resource_limit";
      break;
    }
    if (options.deadline_epoch > 0 &&
        std::time(nullptr) >= options.deadline_epoch) {
      stopped = "deadline";
      break;
    }
    if (std::chrono::duration<double>(now - last_status).count() >= 60) {
      WriteStatus(options, state, "training");
      last_status = now;
    }
    if (std::chrono::duration<double>(now - last_checkpoint).count() >=
        options.checkpoint_interval) {
      SaveCheckpoint(options.checkpoint, state);
      WriteStatus(options, state, "checkpointed");
      last_checkpoint = Clock::now();
      batch_start = last_checkpoint;
    }
  }
  SaveCheckpoint(options.checkpoint, state);
  WriteStatus(options, state, stopped == "complete" ? "trained" : stopped);
  if (options.final_eval) {
    EvaluateBoth(game, *state.solver, options);
    WriteStatus(options, state, stopped);
  }
  return stopped == "complete";
}

void Play(const Options& options, const os::Game& game, const Solver& solver) {
  auto policy = solver.AveragePolicy();
  std::mt19937 rng(options.seed_explicit ? options.seed : std::random_device{}());
  int human_seat = options.human_seat;
  for (int hand = 1;; ++hand) {
    auto state = game.NewInitialState();
    std::cout << "Hand " << hand << ". You are "
              << (human_seat == 1 ? "button/small blind" : "big blind")
              << ".\n";
    while (!state->IsTerminal()) {
      if (state->IsChanceNode()) {
        DealChance(*state, rng);
        continue;
      }
      const int player = state->CurrentPlayer();
      if (player != human_seat) {
        const os::Action action = PolicyAction(*state, *policy, rng);
        std::cout << "Bot: " << state->ActionToString(player, action) << '\n';
        state->ApplyAction(action);
        continue;
      }
      const auto& wrapped =
          static_cast<const os::hu_limit_bot::AbstractState&>(*state);
      const auto& poker = static_cast<const os::universal_poker::UniversalPokerState&>(
          wrapped.GetWrappedState());
      const auto& acpc = poker.acpc_state();
      os::universal_poker::logic::CardSet mine;
      os::universal_poker::logic::CardSet board;
      for (int i = 0; i < 2; ++i) mine.AddCard(acpc.hole_cards(player, i));
      constexpr int kBoardCounts[] = {0, 3, 4, 5};
      for (int i = 0; i < kBoardCounts[acpc.GetRound()]; ++i) {
        board.AddCard(acpc.board_cards(i));
      }
      std::cout << "Round " << acpc.GetRound() << " | Your cards "
                << mine.ToString() << " | Board " << board.ToString()
                << " | Pot " << acpc.TotalSpent() << " | To call "
                << acpc.MaxSpend() - acpc.CurrentSpent(player) << '\n';
      const auto legal = state->LegalActions();
      std::cout << "Actions:";
      for (os::Action action : legal) {
        std::cout << ' ' << action << '=' << state->ActionToString(player, action);
      }
      std::cout << "\nChoose action number (q to quit): " << std::flush;
      std::string text;
      if (!(std::cin >> text) || text == "q") return;
      int choice = -1;
      try {
        size_t consumed = 0;
        choice = std::stoi(text, &consumed);
        if (consumed != text.size()) choice = -1;
      } catch (const std::exception&) {
        choice = -1;
      }
      if (std::find(legal.begin(), legal.end(), choice) == legal.end()) {
        std::cout << "Illegal action\n";
        continue;
      }
      state->ApplyAction(choice);
    }
    std::cout << "Hand over. Your chip return: "
              << state->Returns().at(human_seat) << '\n';
    std::cout << "Another hand? [y/N] " << std::flush;
    std::string answer;
    if (!(std::cin >> answer) || (answer != "y" && answer != "Y")) return;
    human_seat = 1 - human_seat;
  }
}

void PrintHelp() {
  std::cout << "hu_limit_bot train|eval|play [--name=value]\n"
               "  --checkpoint=PATH          Checkpoint file (auto resume in train)\n"
               "  --status=PATH              Live status file\n"
               "  --seconds=43200            Target active training seconds\n"
               "  --deadline_epoch=UNIX      Stop before this wall clock deadline\n"
               "  --checkpoint_interval=600 Seconds between saves\n"
               "  --hands=1000               Hands per evaluation baseline\n"
               "  --final_eval=1             Evaluate after train (runner uses 0)\n"
               "  --human_seat=1             Seat for play mode (0 or 1)\n";
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Options options = ParseOptions(argc, argv);
    if (options.mode == "help" || options.mode == "--help" ||
        options.mode.empty()) {
      PrintHelp();
      return 0;
    }
    auto game = os::LoadGame(os::hu_limit_bot::kGameString);
    if (options.mode == "train") {
      if (!Train(options, *game)) return 2;
    } else if (options.mode == "eval" || options.mode == "play") {
      TrainingState state = ReadCheckpoint(options.checkpoint, *game);
      if (options.mode == "eval") EvaluateBoth(*game, *state.solver, options);
      else Play(options, *game, *state.solver);
    } else {
      throw std::invalid_argument("unknown mode " + options.mode);
    }
  } catch (const std::exception& error) {
    std::cerr << "hu_limit_bot: " << error.what() << '\n';
    return 1;
  }
}
