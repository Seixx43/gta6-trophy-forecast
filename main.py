"""gta6-trophy-rarity-forecast-01 entry: growth-curve forecaster (Python 3.13, stdlib only).

Reads every <id>.in in $MAKEFUN_INPUT_DIR (else /input) and writes <id>.out to
$MAKEFUN_OUTPUT_DIR (else /output): {"trophies": [{"id": n, "q": [q05..q95]}...]}.

The data-generating process this models:
  - Exophase's page is stale between its own periodic refreshes. A read shows
    earned_i / players_tracked as of the last refresh before it, so a refresh
    moves every rate down at once (the tracked count jumps) except a 100% one.
  - Earned counts grow in time since release like a saturating curve; over week
    one a power law fits them well: e_i ~ a_i t^b_i and P ~ c t^d, so each
    rate follows rate_i ~ A_i t^(b_i - d).
  - Median: anchored on the last refresh snapshot's rate, extrapolated by the
    fitted log-rate slope m_i = b_i - d over ln(t_eff / t_anchor), where the
    target is pulled back by half the observed refresh cadence (the answering
    read shows the page as of Exophase's last refresh before it).
  - Spread (logit scale, analytic - no randomness, byte-identical reruns):
    slope-extrapolation error (fit se plus a model-misspecification term),
    per-refresh shock noise seen in the window, small-earned-count noise, and
    snapshot staleness; floored and capped.
  - No refresh visible in the whole window (the launch-night shape of the
    practice cases): the forecast is a launch-night prior instead - a refresh
    lands before the target with the tracked count jumping ~6x (638 -> 3,868
    observed) while earned counts grow less. Per trophy y/r_last is
    lognormal(ln -1.25, sigma) truncated to [-1.90, 0.10] (earned counts are
    cumulative, and the refresh moves rates down); 100% stays 100%.
"""

import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from statistics import NormalDist, median

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = (0.05, 0.25, 0.5, 0.75, 0.95)
Z = tuple(NormalDist().inv_cdf(t) for t in LEVELS)

# Launch-night prior: ln(y / r_last) while no refresh is visible in the window.
LN_MU, LN_SIGMA, LN_LO, LN_HI = -1.25, 0.57, -1.90, 0.10

# Growth-curve regime constants.
KAPPA = 0.18            # log-rate slope model error per ln(t) unit (mis-specification)
S_MIN, S_MAX = 0.12, 2.0
GAP_LN_MAX = 4.0        # cap on ln(t_eff / t_anchor) fed into the spread
D_PRIOR, SE_D_PRIOR = 0.6, 0.35   # players-tracked exponent when too few snapshots
SE_M_NODATA = 0.35                # slope se when a trophy has too few usable points
T_MIN_FIT = 0.25        # hours since release a snapshot needs to enter a fit
E_MIN_FIT = 0.75        # earned-count floor for a usable earned-space fit point
MAX_CADENCE_H = 48.0


def clamp(x, lo, hi):
    return lo if x < lo else (hi if x > hi else x)


def logit_clamped(pct):
    p = clamp(pct, 0.01, 99.99)
    return math.log(p / (100.0 - p))


def sigmoid_pct(x):
    if x >= 0:
        return 100.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return 100.0 * e / (1.0 + e)


def round4(x):
    """Half-up to 4 decimals; a whole number comes back as an int (matches the starter kit's output format)."""
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if value.is_integer() else value


def parse_dt(s):
    s = s.strip()
    if s.endswith(("Z", "z")):
        s = s[:-1] + "+00:00"
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def loglog_fit(pts):
    """Trimmed least squares of ln y on ln t.

    Returns (intercept, slope, se_slope, resid_sd, resid_sd_recent) or None,
    where resid_sd_recent uses the last few residuals (recent refresh shocks
    are the ones like the future's).
    """
    pts = [(t, y) for t, y in pts if t > 0 and y > 0 and math.isfinite(t) and math.isfinite(y)]
    if len(pts) < 3:
        return None

    def ls(ps):
        n = len(ps)
        xs = [math.log(t) for t, _ in ps]
        ys = [math.log(y) for _, y in ps]
        mx = sum(xs) / n
        my = sum(ys) / n
        sxx = sum((x - mx) ** 2 for x in xs)
        if sxx <= 1e-12:
            return None
        slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
        intercept = my - slope * mx
        res = [y - (intercept + slope * x) for x, y in zip(xs, ys)]
        var = sum(r * r for r in res) / (n - 2)
        se = math.sqrt(var / sxx) if var > 0 else 0.0
        return intercept, slope, se, math.sqrt(var), res

    fit = ls(pts)
    if fit is None:
        return None
    a, b, se_b, sd, res = fit
    if len(pts) >= 6 and sd > 0:
        kept = [p for p in pts if abs(math.log(p[1]) - (a + b * math.log(p[0]))) <= 2.5 * sd]
        if 3 <= len(kept) < len(pts):
            refit = ls(kept)
            if refit is not None:
                fit = refit
    a, b, se_b, sd, res = fit
    tail = res[-6:]
    sd_recent = math.sqrt(sum(r * r for r in tail) / len(tail)) if len(tail) >= 3 else sd
    return a, b, se_b, sd, sd_recent


def extract_snapshots(history, release):
    """Reads as (t_hours, players, rates) plus one snapshot per distinct page state.

    A snapshot is the first read showing a new state; its state was produced by
    a refresh inside the previous read interval, so the midpoint of that
    interval is the least-biased timestamp for it. The first read's state age
    is unknown, so it keeps its own time.
    """
    reads = []
    for h in history:
        t = (parse_dt(h["at"]) - release).total_seconds() / 3600.0
        reads.append((t, h.get("players_tracked"), list(h.get("rates") or [])))
    snaps = []
    prev = None
    for t, players, rates in reads:
        if prev is None:
            changed = True
        else:
            changed = (
                players != prev[1]
                or len(rates) != len(prev[2])
                or any(
                    (a is None) != (b is None) or (a is not None and b is not None and a != b)
                    for a, b in zip(rates, prev[2])
                )
            )
        if changed:
            t_attr = t if prev is None else 0.5 * (t + prev[0])
            snaps.append((t_attr, t, players, rates))
        prev = (t, players, rates)
    return reads, snaps


def trophy_rates(reads, i):
    """Every non-null rate the history shows for trophy i, oldest first."""
    return [rs[i] for (_, _, rs) in reads if i < len(rs) and rs[i] is not None]


def launch_night_rows(case, reads):
    rows = []
    for i, tr in enumerate(case["trophies"]):
        seen = trophy_rates(reads, i)
        r = seen[-1] if seen else None
        if r is None:
            rows.append({"id": tr["id"], "q": [0, 0, 0, 0, 0]})
            continue
        if r >= 99.995:
            rows.append({"id": tr["id"], "q": [99.9, 100, 100, 100, 100]})
            continue
        if r <= 0.004:
            rows.append({"id": tr["id"], "q": [0, 0, 0, 0.01, 0.05]})
            continue
        players = reads[-1][1] or 0
        e_last = r * players / 100.0
        # A tiny earned count makes the answering rate itself noisy (one person
        # moves a 0.1% rate by a third), so widen the prior for those trophies.
        s_count = clamp(1.0 / math.sqrt(max(e_last * 1.8, 0.5)), 0.0, 1.2)
        nd = NormalDist(mu=LN_MU, sigma=math.sqrt(LN_SIGMA ** 2 + s_count ** 2))
        lo_mass, hi_mass = nd.cdf(LN_LO), nd.cdf(LN_HI)
        span = hi_mass - lo_mass
        q = []
        for tau in LEVELS:
            u = clamp(lo_mass + tau * span, 1e-9, 1.0 - 1e-9)
            lnrho = clamp(nd.inv_cdf(u), -3.5, 0.6)
            q.append(round4(clamp(r * math.exp(lnrho), 0.005, 99.5)))
        rows.append({"id": tr["id"], "q": sorted(q)})
    return {"trophies": rows}


def growth_rows(case, reads, snaps, T_h):
    n_tr = len(case["trophies"])

    gaps = [snaps[j + 1][1] - snaps[j][1] for j in range(len(snaps) - 1)]
    cadence = clamp(median(gaps), 0.5, MAX_CADENCE_H) if gaps else 4.0

    p_pts = [
        (s[0], s[2]) for s in snaps if s[0] >= T_MIN_FIT and isinstance(s[2], (int, float)) and s[2] >= 1
    ]
    f_p = loglog_fit(p_pts)
    if f_p is not None:
        c_p, d_exp, se_d = f_p[0], f_p[1], f_p[2]
    else:
        d_exp, se_d = D_PRIOR, SE_D_PRIOR

    t_a, t_a_raw, p_a, _ = snaps[-1]
    t_a_eff = max(t_a, 0.1)
    t_eff = max(min(T_h - 0.5 * cadence, T_h), t_a_eff + 1e-6)
    gap_ln = clamp(math.log(t_eff / t_a_eff), 0.0, GAP_LN_MAX)
    k_future = clamp((T_h - t_a_raw) / cadence, 0.0, 3.0)
    horizon_days = case["horizon_minutes"] / 1440.0
    denom_t = max(T_h - 0.5 * cadence, 0.5 * T_h)

    rows = []
    for i, tr in enumerate(case["trophies"]):
        seen = trophy_rates(reads, i)
        r_last = seen[-1] if seen else None
        if r_last is None:
            rows.append({"id": tr["id"], "q": [0, 0, 0, 0, 0]})
            continue
        if r_last >= 99.995 and min(seen) >= 99.5:
            rows.append({"id": tr["id"], "q": [99.9, 100, 100, 100, 100]})
            continue
        if r_last <= 0.004:
            rows.append({"id": tr["id"], "q": [0, 0, 0, 0.01, 0.05]})
            continue

        e_pts = []
        r_pts = []
        for t_attr, _t_raw, players, rates in snaps:
            if t_attr < T_MIN_FIT or i >= len(rates) or rates[i] is None or not players:
                continue
            rate = rates[i]
            r_pts.append((t_attr, rate))
            e = rate * players / 100.0
            if e >= E_MIN_FIT:
                e_pts.append((t_attr, e))

        f_i = loglog_fit(e_pts)
        if f_i is not None:
            a_i, b_i, se_b, sd_i, sd_recent = f_i
            m = b_i - d_exp
            se_m = math.hypot(se_b, se_d)
            if f_p is not None:
                # If the anchor snapshot sits far off the fitted level curve, the
                # fitted slope is suspect: forecast flat from the anchor instead.
                arg = clamp((a_i - c_p) + (b_i - d_exp) * math.log(t_a_eff), -12.0, 5.0)
                y_fit_a = 100.0 * math.exp(arg)
                if abs(logit_clamped(r_last) - logit_clamped(clamp(y_fit_a, 0.005, 100.0))) > 0.75:
                    m, se_m = 0.0, max(se_m, SE_M_NODATA)
        else:
            f_r = loglog_fit([(t, r) for t, r in r_pts if r >= 0.01])
            if f_r is not None:
                _a, m, se_m, _sd, sd_recent = f_r
                sd_i = 0.05
            else:
                m, se_m, sd_i, sd_recent = 0.0, SE_M_NODATA, 0.05, 0.05
        m = clamp(m, -1.2, 1.2)

        ratio = math.exp(clamp(m * gap_ln, -2.0, 2.0))
        y_med = clamp(r_last * ratio, 0.005, 100.0)

        s_slope = gap_ln * math.sqrt(se_m ** 2 + KAPPA ** 2)
        e_anchor = (r_last * p_a / 100.0) if p_a else 0.0
        b_use = f_i[1] if f_i is not None else (d_exp if f_p is not None else D_PRIOR)
        e_t = max(e_anchor * (t_eff / t_a_eff) ** b_use, 0.25) if e_anchor > 0 else 0.25
        s_count = clamp(1.0 / math.sqrt(e_t), 0.0, 1.2)
        s_shock = clamp(sd_recent, 0.01, 0.6) * math.sqrt(1.0 + k_future)
        s_stale = abs(m) * abs(math.log(T_h / denom_t)) if T_h > 0 else 0.0
        s = math.sqrt(s_slope ** 2 + s_count ** 2 + s_shock ** 2 + s_stale ** 2)
        if len(e_pts) < 2:
            s = max(s, 0.5 * math.sqrt(horizon_days))
        s = clamp(s, S_MIN, S_MAX)

        level = logit_clamped(y_med)
        q = [round4(clamp(sigmoid_pct(level + z * s), 0.0, 100.0)) for z in Z]
        rows.append({"id": tr["id"], "q": sorted(q)})
    return {"trophies": rows}


def forecast_case(case):
    history = case.get("history") or []
    if not history:
        raise ValueError(f"{case.get('id')}: no history")
    release = parse_dt(case["game"]["release_date"] + "T00:00:00Z")
    reads, snaps = extract_snapshots(history, release)
    if len(snaps) <= 1:
        return launch_night_rows(case, reads)
    target_h = (parse_dt(case["target"]) - release).total_seconds() / 3600.0
    return growth_rows(case, reads, snaps, target_h)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        out = forecast_case(case)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(out, separators=(",", ":")) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
