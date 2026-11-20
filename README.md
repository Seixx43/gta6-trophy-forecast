# gta6-trophy-forecast

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/Seixx43/gta6-trophy-forecast/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/Seixx43/gta6-trophy-forecast/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Forecasts each Grand Theft Auto VI PlayStation 5 trophy's Exophase earn rate at a future target
as five quantiles (5/25/50/75/95%), from the house's recorded page reads. Robust per-trophy
rate curves in tracked-share space, players-tracked extrapolation, and a refresh-timing
mixture spread. Python 3.13 stdlib only, deterministic, runs in well under a second per case.

## Model

- Per state, reconstruct earned counts exactly: rates print to 2 decimals, so the count is
  the integer consistent with round(rate/100 x players, 2); recovered rates feed every fit.
- Growth regime (several page states): rates are fit in tracked-share space
  u = P_first / P(t) - linear r = theta + (h - theta) x u and power law, both by
  median-of-slopes, picked or hedged 50/50 by robust L1; players_tracked increments get
  their own saturating-vs-power fit. The forecast mixes over the answer state's age under
  a Gamma-Poisson refresh hazard fit to the observed gaps (Lomax predictive, no-refresh
  atom at the horizon).
- Launch regime (one page state): refresh-timing atom plus a J^(gamma-1) rate jump across
  the tracked-count sync (J ~ 6.1x observed launch night, gamma ~ 0.30, per-trophy
  residual).
- Spread: per-component analytic quadrature of fit-residual, u-extrapolation, count-noise
  and players-forecast error; ~0.9 coverage in synthetic self-tests.

## Run

    MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py

MIT licensed. Built for the make.fun bounty gta6-trophy-rarity-forecast-01.
