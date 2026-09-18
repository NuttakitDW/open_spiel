// Standalone poker-arena v1 adapter for the trained HU fixed-limit policy.
// The release build embeds the exported model in its read-only data segment.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <iterator>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "nlohmann/json.hpp"
#include "open_spiel/games/universal_poker/acpc/project_acpc_server/evalHandTables"

using Json = nlohmann::json;

#ifdef ARENA_EMBED_MODEL
extern "C" const unsigned char arena_model_begin[];
extern "C" const unsigned char arena_model_end[];
#endif

namespace {

struct Model {
  std::vector<unsigned char> bytes;
  std::unordered_map<std::string_view, std::array<double, 3>> policy;

  static uint16_t U16(const unsigned char*& p, const unsigned char* end) {
    if (end - p < 2) throw std::runtime_error("truncated model");
    uint16_t v = uint16_t(p[0]) | uint16_t(p[1]) << 8;
    p += 2;
    return v;
  }
  static uint32_t U32(const unsigned char*& p, const unsigned char* end) {
    if (end - p < 4) throw std::runtime_error("truncated model");
    uint32_t v = uint32_t(p[0]) | uint32_t(p[1]) << 8 |
                 uint32_t(p[2]) << 16 | uint32_t(p[3]) << 24;
    p += 4;
    return v;
  }
  static double F64(const unsigned char*& p, const unsigned char* end) {
    if (end - p < 8) throw std::runtime_error("truncated model");
    uint64_t bits = 0;
    for (int shift = 0; shift < 64; shift += 8) bits |= uint64_t(*p++) << shift;
    double value;
    std::memcpy(&value, &bits, sizeof(value));
    if (!std::isfinite(value) || value < 0 || value > 1)
      throw std::runtime_error("invalid model probability");
    return value;
  }
  void Load(const std::string& path) {
#ifdef ARENA_EMBED_MODEL
    if (!path.empty()) throw std::runtime_error("embedded model build takes no model path");
    bytes.assign(arena_model_begin, arena_model_end);
#else
    if (path.empty()) throw std::runtime_error("usage: arena_hulhe_bot --model=FILE");
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot open model");
    bytes.assign(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
#endif
    const unsigned char* p = bytes.data();
    const unsigned char* end = p + bytes.size();
    if (end - p < 12 || std::memcmp(p, "HULHEP1\0", 8) != 0)
      throw std::runtime_error("invalid model magic");
    p += 8;
    const uint32_t count = U32(p, end);
    if (!count || count > 500000) throw std::runtime_error("invalid model count");
    policy.reserve(count);
    std::string_view previous;
    for (uint32_t i = 0; i < count; ++i) {
      const uint16_t length = U16(p, end);
      if (!length || end - p < length) throw std::runtime_error("invalid model key");
      const std::string_view key(reinterpret_cast<const char*>(p), length);
      p += length;
      if (i && !(previous < key)) throw std::runtime_error("unsorted model keys");
      previous = key;
      std::array<double, 3> weights{F64(p, end), F64(p, end), F64(p, end)};
      const double sum = weights[0] + weights[1] + weights[2];
      if (sum < 0.999999 || sum > 1.000001)
        throw std::runtime_error("invalid model probability sum");
      policy.emplace(key, weights);
    }
    if (p != end) throw std::runtime_error("trailing model bytes");
  }
};

int Card(const std::string& text) {
  static constexpr std::string_view ranks = "23456789TJQKA";
  static constexpr std::string_view suits = "cdhs";
  if (text.size() != 2) throw std::runtime_error("invalid card");
  const size_t rank = ranks.find(text[0]);
  const size_t suit = suits.find(text[1]);
  if (rank == std::string_view::npos || suit == std::string_view::npos)
    throw std::runtime_error("invalid card");
  return static_cast<int>(rank * 4 + suit);
}

int PreflopBucket(std::array<int, 2> hole) {
  const int a = hole[0] / 4, b = hole[1] / 4;
  if (a == b) return a;
  const int high = std::max(a, b), low = std::min(a, b);
  return (hole[0] % 4 == hole[1] % 4 ? 13 : 91) + high * (high - 1) / 2 + low;
}

uint32_t VisibleSeed(const std::array<int, 2>& hole,
                     const std::vector<int>& board, int round) {
  uint32_t hash = 2166136261u;
  for (int card : hole) hash = (hash ^ uint32_t(card + 1)) * 16777619u;
  for (int card : board) hash = (hash ^ uint32_t(card + 1)) * 16777619u;
  return (hash ^ uint32_t(round + 1)) * 16777619u;
}

// libc++'s uniform_int_distribution for a 32-bit MT engine. Its low-bit mask
// and rejection sequence must match the training machine exactly.
int PortableUniform(std::mt19937& rng, int lower, int upper) {
  const uint32_t n = uint32_t(upper - lower + 1);
  if (n == 1) return lower;
  uint32_t power = 1;
  while (power < n) power <<= 1;
  const uint32_t mask = power - 1;
  uint32_t u;
  do { u = rng() & mask; } while (u >= n);
  return lower + static_cast<int>(u);
}

void AddCard(Cardset* cards, int card) {
  cards->bySuit[card % 4] |= uint16_t(1u << (card / 4));
}

double Equity(std::array<int, 2> hole, std::vector<int> board, int round) {
  std::sort(hole.begin(), hole.end());
  std::sort(board.begin(), board.end());
  std::array<bool, 52> used{};
  for (int card : hole) {
    if (card < 0 || card >= 52 || used[card]) throw std::runtime_error("invalid hole cards");
    used[card] = true;
  }
  for (int card : board) {
    if (card < 0 || card >= 52 || used[card]) throw std::runtime_error("invalid board cards");
    used[card] = true;
  }
  std::vector<int> remaining;
  remaining.reserve(50 - board.size());
  for (int card = 0; card < 52; ++card) if (!used[card]) remaining.push_back(card);
  std::mt19937 rng(VisibleSeed(hole, board, round));
  const int draw_count = 7 - static_cast<int>(board.size());
  double wins = 0;
  for (int sample = 0; sample < 128; ++sample) {
    for (int i = 0; i < draw_count; ++i) {
      const int index = PortableUniform(rng, i, static_cast<int>(remaining.size()) - 1);
      std::swap(remaining[i], remaining[index]);
    }
    Cardset mine = emptyCardset(), theirs = emptyCardset();
    AddCard(&mine, hole[0]);
    AddCard(&mine, hole[1]);
    AddCard(&theirs, remaining[0]);
    AddCard(&theirs, remaining[1]);
    for (int card : board) { AddCard(&mine, card); AddCard(&theirs, card); }
    for (int i = 2; i < draw_count; ++i) {
      AddCard(&mine, remaining[i]);
      AddCard(&theirs, remaining[i]);
    }
    const int a = rankCardset(mine), b = rankCardset(theirs);
    wins += a > b ? 1.0 : a == b ? 0.5 : 0.0;
  }
  return wins / 128.0;
}

int Bucket(const std::array<int, 2>& hole, const std::vector<int>& board, int round) {
  if (round == 0) return PreflopBucket(hole);
  return std::min(31, static_cast<int>(Equity(hole, board, round) * 32));
}

struct Decision {
  bool fold = false;
  bool check = false;
  bool call = false;
  bool bet = false;
  bool raise = false;
  uint64_t call_amount = 0;
  uint64_t bet_to = 0;
  uint64_t raise_to = 0;
  int Mask() const { return (fold ? 1 : 0) | (check || call ? 2 : 0) |
                             (bet || raise ? 4 : 0); }
};

bool Has(const Json& object, const char* key) {
  return object.is_object() && object.contains(key) && !object.at(key).is_null();
}
uint64_t U64(const Json& object, const char* key) {
  if (!Has(object, key)) throw std::runtime_error(std::string("missing ") + key);
  if (!object.at(key).is_number_unsigned())
    throw std::runtime_error(std::string("invalid unsigned integer ") + key);
  return object.at(key).get<uint64_t>();
}
int Seat(const Json& object) {
  const uint64_t value = U64(object, "seat");
  if (value > 1) throw std::runtime_error("invalid seat");
  return static_cast<int>(value);
}
std::string String(const Json& object, const char* key) {
  if (!Has(object, key)) throw std::runtime_error(std::string("missing ") + key);
  return object.at(key).get<std::string>();
}
Decision ReadDecision(const Json& value) {
  if (String(value, "kind") != "wager") throw std::runtime_error("unsupported decision kind");
  Decision d;
  d.fold = value.value("fold", false);
  d.check = value.value("check", false);
  d.call = Has(value, "call");
  d.bet = Has(value, "bet");
  d.raise = Has(value, "raise");
  if (d.call) d.call_amount = U64(value, "call");
  if (d.bet) d.bet_to = U64(value.at("bet"), "min_to");
  if (d.raise) d.raise_to = U64(value.at("raise"), "min_to");
  if ((!d.check && !d.call) || (d.check && d.call) || (d.bet && d.raise))
    throw std::runtime_error("invalid wager decision");
  return d;
}

struct Hand {
  int seat = -1;
  int round = 0;
  std::array<int, 2> hole{-1, -1};
  std::vector<int> board;
  std::array<std::string, 4> sequence;
  std::array<uint64_t, 2> commit{};
  uint64_t pot = 0;
  bool out_of_model = false;
  std::string reason;
  void Incompatible(std::string why) {
    if (!out_of_model) { out_of_model = true; reason = std::move(why); }
  }
};

std::string Key(const Hand& hand, int bucket, int mask) {
  std::string key = "v1|p" + std::to_string(1 - hand.seat) +
                    "|r" + std::to_string(hand.round) + "|s";
  for (int r = 0; r <= hand.round; ++r) {
    if (r) key += '/';
    key += hand.sequence[r];
  }
  return key + "|b" + std::to_string(bucket) + "|m" + std::to_string(mask);
}

struct Bot {
  Model model;
  Hand hand;
  std::mt19937 rng{0xC0FFEEu};
  uint64_t small_blind = 0, big_blind = 0, starting_stack = 0;
  bool compatible = false;
  uint64_t trained_hits = 0, unseen = 0, incompatible = 0, actions = 0;

  void Hello(const Json& message) {
    if (U64(message, "proto") != 1 || String(message, "game_id") != "holdem-fl" ||
        U64(message, "seat_count") != 2 ||
        String(message.at("betting"), "kind") != "fixed-limit" ||
        String(message.at("stakes"), "kind") != "blinds")
      throw std::runtime_error("only v1 heads-up holdem-fl supported");
    const Json& stakes = message.at("stakes");
    small_blind = U64(stakes, "small_blind");
    big_blind = U64(stakes, "big_blind");
    starting_stack = U64(message, "starting_stack");
    compatible = small_blind > 0 && small_blind <= UINT64_MAX / 2 &&
                 big_blind == small_blind * 2 && U64(stakes, "ante") == 0 &&
                 Has(message.at("betting"), "raise_cap") &&
                 U64(message.at("betting"), "raise_cap") == 4 &&
                 big_blind <= UINT64_MAX / 24 &&
                 starting_stack >= 24 * big_blind;
    if (!compatible) std::cerr << "arena_bot: configured stakes/cap/stack outside trained game; using legal fallback\n";
    std::cout << "{\"t\":\"join\"}\n" << std::flush;
  }

  void Event(const Json& event) {
    const std::string tag = String(event, "event");
    if (tag == "hand-start") {
      const auto& stacks = event.at("stacks");
      if (!stacks.is_array() || stacks.size() != 2) throw std::runtime_error("invalid stacks");
      for (const auto& stack : stacks) {
        if (!stack.is_number_unsigned() ||
            (compatible && stack.get<uint64_t>() < 24 * big_blind))
          hand.Incompatible("binding starting stack");
      }
    } else if (tag == "post") {
      const int seat = Seat(event);
      const uint64_t amount = U64(event, "amount");
      const std::string kind = String(event, "kind");
      if (amount > UINT64_MAX - hand.pot) throw std::runtime_error("pot overflow");
      hand.pot += amount;
      if (kind != "ante") {
        if (amount > UINT64_MAX - hand.commit[seat]) throw std::runtime_error("commit overflow");
        hand.commit[seat] += amount;
        if ((kind == "small-blind" && amount != small_blind) ||
            (kind == "big-blind" && amount != big_blind))
          hand.Incompatible("blind amount");
      } else hand.Incompatible("ante");
      if (event.value("all_in", false)) hand.Incompatible("post all-in");
    } else if (tag == "street-start") {
      const uint64_t value = U64(event, "street");
      if (value > 3) throw std::runtime_error("invalid street");
      const int street = static_cast<int>(value);
      if (street < hand.round || street > hand.round + 1)
        throw std::runtime_error("invalid street");
      if (street != 0) hand.commit = {0, 0};
      hand.round = street;
    } else if (tag == "deal-hole") {
      const int seat = Seat(event);
      if (seat == hand.seat) {
        const auto& cards = event.at("cards");
        if (!cards.is_array() || cards.size() != 2) throw std::runtime_error("missing own cards");
        hand.hole = {Card(cards[0].get<std::string>()), Card(cards[1].get<std::string>())};
      }
    } else if (tag == "deal-community") {
      const auto& cards = event.at("cards");
      if (!cards.is_array()) throw std::runtime_error("invalid board event");
      for (const auto& card : cards) hand.board.push_back(Card(card.get<std::string>()));
      if (hand.board.size() > 5) throw std::runtime_error("invalid board length");
    } else if (tag == "acted") {
      const int seat = Seat(event);
      const uint64_t next = U64(event, "street_commit");
      if (next < hand.commit[seat] || next - hand.commit[seat] > UINT64_MAX - hand.pot)
        throw std::runtime_error("invalid commitment");
      const uint64_t delta = next - hand.commit[seat];
      hand.pot += delta;
      const std::string kind = String(event.at("action"), "kind");
      if (kind == "fold") hand.sequence[hand.round] += 'f';
      else if (kind == "check" || kind == "call") hand.sequence[hand.round] += 'c';
      else if (kind == "bet" || kind == "raise") hand.sequence[hand.round] += 'r';
      else { hand.Incompatible("unknown acted action"); return; }
      if (kind == "check" || kind == "fold") {
        if (delta) hand.Incompatible("unexpected no-chip action");
      } else if (kind == "call") {
        const uint64_t expected = std::max(hand.commit[0], hand.commit[1]);
        if (next != expected) hand.Incompatible("short call");
      } else {
        if (hand.round >= 2 && big_blind > UINT64_MAX / 2)
          hand.Incompatible("bet size overflow");
        const uint64_t expected_increment =
            hand.round < 2 ? big_blind :
            big_blind <= UINT64_MAX / 2 ? big_blind * 2 : 0;
        const uint64_t prior_high = std::max(hand.commit[0], hand.commit[1]);
        if (next < prior_high || next - prior_high != expected_increment)
          hand.Incompatible("short or unexpected raise");
      }
      hand.commit[seat] = next;
      if (event.value("all_in", false)) hand.Incompatible("all-in");
    }
    // All unrecognized event tags are deliberately ignored.
  }

  int Sample(const std::array<double, 3>& probabilities, int mask) {
    std::array<double, 3> p{};
    double sum = 0;
    for (int a = 0; a < 3; ++a) if (mask & (1 << a)) sum += p[a] = probabilities[a];
    if (!(sum > 0)) {
      for (int a = 0; a < 3; ++a) if (mask & (1 << a)) sum += p[a] = 1;
    }
    const double draw = std::generate_canonical<double, 53>(rng) * sum;
    double cumulative = 0;
    for (int a = 0; a < 3; ++a) if (mask & (1 << a)) {
      cumulative += p[a];
      if (draw < cumulative) return a;
    }
    for (int a = 2; a >= 0; --a) if (mask & (1 << a)) return a;
    throw std::runtime_error("no legal actions");
  }

  int Fallback(const Decision& decision) {
    // Visible-card equity and pot odds; never consults the actual opponent hand.
    const int mask = decision.Mask();
    if (hand.hole[0] < 0 || hand.hole[1] < 0) return decision.check || decision.call ? 1 : 0;
    const double equity = Equity(hand.hole, hand.board, hand.round);
    if (decision.call) {
      const double price = double(decision.call_amount) /
                           (double(hand.pot) + double(decision.call_amount));
      if (decision.fold && equity + 0.04 < price) return 0;
    }
    if (mask & 4 && equity > (hand.round == 0 ? 0.66 : 0.70)) return 2;
    return 1;
  }

  void Act(const Json& message) {
    if (Seat(message) != hand.seat)
      throw std::runtime_error("act seat mismatch");
    const Decision decision = ReadDecision(message.at("decision"));
    const int mask = decision.Mask();
    if (!(mask & 2)) throw std::runtime_error("wager lacks check/call");
    // Reject finite-stack/all-in bet sizes before indexing the trained table.
    if (compatible && !hand.out_of_model && (decision.bet || decision.raise)) {
      const Json& wager = message.at("decision").at(decision.bet ? "bet" : "raise");
      const uint64_t target = decision.bet ? decision.bet_to : decision.raise_to;
      const uint64_t high = std::max(hand.commit[0], hand.commit[1]);
      const uint64_t increment = hand.round < 2 ? big_blind : 2 * big_blind;
      if (high > UINT64_MAX - increment ||
          U64(wager, "max_to") != target || target != high + increment)
        hand.Incompatible("short or unexpected legal raise");
    }
    if (hand.hole[0] < 0 || hand.hole[1] < 0 || hand.hole[0] == hand.hole[1])
      throw std::runtime_error("missing or duplicate own cards");
    constexpr int board_counts[4] = {0, 3, 4, 5};
    if (hand.board.size() != board_counts[hand.round])
      throw std::runtime_error("incomplete visible board");
    int action;
    if (!compatible || hand.out_of_model) {
      ++incompatible;
      action = Fallback(decision);
    } else {
      const std::string key = Key(hand, Bucket(hand.hole, hand.board, hand.round), mask);
      const auto it = model.policy.find(key);
      if (it == model.policy.end()) {
        ++unseen;
        action = Sample({0, 0, 0}, mask);
      } else {
        ++trained_hits;
        action = Sample(it->second, mask);
      }
    }
    ++actions;
    if (actions == 100 || actions % 1000 == 0) {
      std::cerr << "arena_bot: actions=" << actions << " trained_hits="
                << trained_hits << " unseen=" << unseen
                << " incompatible=" << incompatible << '\n';
    }
    Json output = {{"t", "action"}, {"action", {{"kind", "check"}}}};
    if (action == 0 && decision.fold) output["action"]["kind"] = "fold";
    else if (action == 2 && (decision.bet || decision.raise)) {
      output["action"]["kind"] = decision.bet ? "bet" : "raise";
      output["action"]["to"] = decision.bet ? decision.bet_to : decision.raise_to;
    } else output["action"]["kind"] = decision.check ? "check" : "call";
    std::cout << output.dump() << '\n' << std::flush;
  }
};

void BucketTest() {
  int round, a, b;
  while (std::cin >> round >> a >> b) {
    if (round < 0 || round > 3) throw std::runtime_error("invalid test round");
    std::vector<int> board(round == 0 ? 0 : round + 2);
    for (int& card : board) if (!(std::cin >> card)) throw std::runtime_error("invalid test board");
    std::cout << Bucket({a, b}, board, round) << '\n';
  }
}

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc == 2 && std::string_view(argv[1]) == "--bucket-test") {
      BucketTest();
      return 0;
    }
    std::string path;
    if (argc == 2 && std::string_view(argv[1]).substr(0, 8) == "--model=")
      path = std::string(argv[1] + 8);
    else if (argc != 1) throw std::runtime_error("usage: arena_hulhe_bot [--model=FILE]");
    Bot bot;
    bot.model.Load(path);
    bool joined = false;
    std::string line;
    while (true) {
      line.clear();
      int ch;
      while ((ch = std::cin.get()) != EOF && ch != '\n') {
        if (line.size() == 65536) throw std::runtime_error("oversized protocol line");
        line.push_back(static_cast<char>(ch));
      }
      if (ch == EOF && line.empty()) break;
      if (line.empty() || std::all_of(line.begin(), line.end(), [](unsigned char c) {
            return c == ' ' || c == '\t' || c == '\r';
          })) continue;
      const Json message = Json::parse(line);
      if (!message.is_object() || !Has(message, "t")) throw std::runtime_error("invalid protocol message");
      const std::string tag = String(message, "t");
      if (tag == "hello") {
        if (joined) throw std::runtime_error("duplicate hello");
        bot.Hello(message);
        joined = true;
      } else if (!joined && tag != "joined" && tag != "event" && tag != "act" &&
                 tag != "hand-start" && tag != "hand-end" && tag != "match-end") continue;
      else if (!joined) throw std::runtime_error("message before hello");
      else if (tag == "hand-start") {
        bot.hand = Hand{};
        bot.hand.seat = Seat(message);
      } else if (tag == "event") bot.Event(message.at("ev"));
      else if (tag == "act") bot.Act(message);
      else if (tag == "match-end") break;
      // joined, hand-end, and future tags need no action.
    }
    std::cerr << "arena_bot: actions=" << bot.actions << " trained_hits="
              << bot.trained_hits << " unseen=" << bot.unseen
              << " incompatible=" << bot.incompatible << '\n';
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "arena_bot: " << error.what() << '\n';
    return 1;
  }
}
