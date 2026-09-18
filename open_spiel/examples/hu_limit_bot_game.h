#ifndef OPEN_SPIEL_EXAMPLES_HU_LIMIT_BOT_GAME_H_
#define OPEN_SPIEL_EXAMPLES_HU_LIMIT_BOT_GAME_H_

#include <array>
#include <memory>
#include <string>

#include "open_spiel/game_transforms/game_wrapper.h"

namespace open_spiel::hu_limit_bot {

inline constexpr char kGameString[] =
    "hu_limit_holdem(equity_bins=32,equity_samples=128)";

class AbstractState final : public WrappedState {
 public:
  AbstractState(std::shared_ptr<const Game> game, std::unique_ptr<State> state,
                int equity_bins, int equity_samples);
  AbstractState(const AbstractState& other) = default;

  std::string InformationStateString(Player player) const override;
  double EstimatedEquity(Player player) const;
  void InformationStateTensor(Player player,
                              absl::Span<float> values) const override;
  std::unique_ptr<State> Clone() const override;

 protected:
  void DoApplyAction(Action action) override;

 private:
  int CardBucket(Player player) const;

  int equity_bins_;
  int equity_samples_;
  mutable std::array<int, 2> cached_bucket_{{-1, -1}};
  mutable std::array<double, 2> cached_equity_{{-1.0, -1.0}};
};

class AbstractGame final : public WrappedGame {
 public:
  AbstractGame(std::shared_ptr<const Game> game, GameType game_type,
               GameParameters params);

  std::unique_ptr<State> NewInitialState() const override;
  std::vector<int> InformationStateTensorShape() const override { return {}; }

 private:
  int equity_bins_;
  int equity_samples_;
};

}  // namespace open_spiel::hu_limit_bot

#endif  // OPEN_SPIEL_EXAMPLES_HU_LIMIT_BOT_GAME_H_
