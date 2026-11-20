# gta6-trophy-forecast

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/Seixx43/gta6-trophy-forecast/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/Seixx43/gta6-trophy-forecast/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Forecasts each Grand Theft Auto VI PlayStation 5 trophy's Exophase earn rate at a future target
as five quantiles (5/25/50/75/95%), from the house's recorded page reads. Saturating growth
curve fit in earned-count space, players-tracked extrapolation, and a refresh-aware logit-scale
spread. Python 3.13 stdlib only, deterministic, runs in well under a second per case.

## Model

- Per trophy, reconstruct earned counts from each read: earned = rate x players_tracked / 100.
- Fit a power-law saturation curve in earned-count space over the observed window.
- Extrapolate players_tracked growth the same way; rate_target = earned_target / players_target.
- Quantile spread: logit-scale quadrature of slope-extrapolation error, count noise, and
  Exophase refresh-jump risk estimated from observed refreshes in the case history.

## Run

    MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py

MIT licensed. Built for the make.fun bounty gta6-trophy-rarity-forecast-01.
