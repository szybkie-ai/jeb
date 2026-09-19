
## Price-movement labels and the derived strategy (the private trading model)

`hindsight.py` (default `--labels excursions`) makes a decision point of every one-minute bar that follows a significant
move: at least `--trigger-atr` (0.75) hourly ATRs over the last 15 minutes, no spacing and no cooldown, so bursts yield
runs of consecutive points. At each point the model is asked what the price does over the next 8 hours, in units of
the 1-hour ATR (14 bars): which 2-ATR move comes first (`first_move`: up, down, neither), how far the highest and the
lowest prices get (`up_move`, `down_move`, six levels) and where the price ends up (`close_move`, seven signed levels).
The targets are the realised levels, one-hot, so across many similar states the model learns P(level | state), a
conditional distribution over the future rather than a softmax over one path. The state is market-only (instrument,
weekday, hour, session, recent moves in percent and in ATR units, ranges, the four charts): no account, no position.
`level_values.py` derives the representative value of each level from the labels (`data/level_values.json`).

The strategy lives in code (`jeb_trading/excursions.py`, `Strategy`): side from the first-move margin and the sign of
the expected close, the tightest stop whose adverse-excursion probability is at most 0.35, the farthest target the
favourable excursion reaches with probability at least 0.45, an ordering-aware win probability, and a trade only when the
implied expected R clears `theta` (0.2). In the gym (`run.py --policy excursion`) the forecaster is asked on the same
triggers whether or not a position is open; a position is closed at the horizon, or when a fresh forecast turns
against it by the entry margin, otherwise it runs to its stop or target. `--theta`, `--delta`, `--risk` and `--horizon`
tune the layer without retraining. Holdout label files (`hindsight_ex_holdout_*.jsonl`) give the held-out
cross-entropy against the uniform and marginal baselines, the cleaner signal test next to the equity curve.
