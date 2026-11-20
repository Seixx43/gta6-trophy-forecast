# gta6-trophy-rarity-forecast-01 — growth-curve forecaster

Forecasts, for every Grand Theft Auto VI (PS5) trophy, its earn rate on
Exophase at a case's target time, as five quantiles (0.05, 0.25, 0.5, 0.75,
0.95). Built for the make.fun bounty `gta6-trophy-rarity-forecast-01`.

## Run

Python 3.13, standard library only, no network, deterministic (identical bytes
on every run):

```sh
mkdir -p out
MAKEFUN_INPUT_DIR=/path/to/input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

It reads every `*.in` in the input directory and writes `<id>.out` next to the
output directory: `{"trophies": [{"id": 1, "q": [q05, q25, q50, q75, q95]}, ...]}`.

## Model

The data-generating process the model assumes:

- **Stale page, jump refreshes.** Exophase serves a trophy page that only
  changes at its own periodic refreshes: between them a read shows the same
  snapshot (on launch night, 638 players tracked for ~4.2 hours, then one
  refresh to 3,868). Every refresh moves every rate *down* at once (the
  tracked denominator jumps) except a 100% trophy.
- **Saturating growth.** Earned counts grow in time since release like a
  saturating curve, and the tracked-player count grows fast at launch and
  decelerates. Over week one both are fit well by power laws:
  `earned_i ~ a_i t^b_i`, `players ~ c t^d`, so each rate follows
  `rate_i ~ A_i t^(b_i - d)`.

The program first dedupes the case's reads into *snapshots* — the first read
showing each distinct page state, timestamped at the midpoint of the read
interval the refresh landed in (the first read keeps its own time; its state's
age is unknown).

**Growth-curve regime** (a refresh is visible in the window — the final's
shape): per trophy, least squares of `ln earned` on `ln t` over snapshots (one
residual-trimming pass; snapshots before 0.25 h after release excluded, since
they predate launch), and the same for `ln players`. The median is anchored on
the last snapshot's rate and extrapolated by the fitted slope difference
`m_i = b_i − d` over `ln(t_eff / t_anchor)`, where the target is pulled back
by half the observed refresh cadence — the answering read shows the page as of
Exophase's last refresh before it, not the instant itself. Anchoring on the
last snapshot (rather than extrapolating both fitted levels) removes intercept
error; a last snapshot far off the fitted level (> 0.75 logit) marks the slope
suspect and the forecast goes flat instead. Fallbacks: with too few usable
earned points the slope comes from a direct `ln rate` fit, else from a flat
no-trend prior.

**Spread** (logit scale, analytic, no randomness): the quadrature of

- slope-extrapolation error: `Δln t × sqrt(se(m_i)² + κ²)` with κ = 0.18 of
  model-misspecification per `ln t` unit (a power-law slope fit over ~2 days
  need not hold for the next ~4.5),
- per-refresh shock noise: the recent residual scale around the fit, times
  `sqrt(1 + expected refreshes before the answer)` (capped at 3),
- small-count noise: `1/sqrt(earned at target)` — one person moves a 0.1%
  rate by a third, so rare trophies are genuinely noisy in logit terms,
- snapshot-staleness: the answering snapshot can be up to one cadence older
  than the target state,

floored at 0.12 and capped at 2.0 logit; trophies with too little data fall
back to the house-baseline width `0.5·sqrt(horizon_days)`. Quantiles are
`100 / (1 + exp(-(logit(median) + z_tau·s)))`, rounded half-up to 4 decimals.

**Launch-night regime** (no refresh visible anywhere in the window — the
practice cases' shape, where nothing can be fit): per trophy, the refresh
ratio `y / r_last` is drawn from a lognormal prior `exp(N(ln −1.25, σ))`
truncated to `[−1.90, 0.10]` — encoding the observed launch-night pattern
(tracked count jumps ~6×, 638 → 3,868, while earned counts grow less; earned
counts are cumulative so the ratio cannot fall below `1/J`, and the rules
observe every non-100% rate moved down) widened by small-count noise. A 100%
trophy stays 100% (with a 99.9 lower hedge); a 0.00 trophy gets a ladder
hugging the 0.01 logit clamp, where upward hedges are nearly free.

**Edge cases:** 100% throughout → `[99.9, 100, 100, 100, 100]`; 0.00 last
rate → `[0, 0, 0, 0.01, 0.05]`; null rates are skipped per trophy; a trophy
never seen → all zeros (the scoring clamp makes 0.00 and 0.01 identical).

## Verification

- Runs the practice cases in ~0.04 s (limit: 8 s), byte-identical across runs.
- Output shape validated: one row per trophy id, five finite quantiles in
  [0, 100], nondecreasing.
- The growth regime is exercised on synthetic f00-shaped cases generated under
  the assumed process (stale page between refreshes, power-law growth, 2-decimal
  rounding, injects: 100% trophy, 0.00 trophy, nulls, short history, frequent
  and rare refresh cadences). Across variants the model scores 0.04–0.10 mean
  pinball loss against the house baseline's ~0.27, with 54–55/55 coverage.

## License

MIT (see `LICENSE`).
