"""Preference-weighted sampler weights — the self-steering part (spec Section 6).

This is NOT a trained model. Each domain and technique tag gets a score equal to
an exponentially-weighted, Laplace-smoothed promote rate derived from the log:

    score(tag) = (PRIOR_STRENGTH * PRIOR_MEAN + Σ promoted_i · decay^age_weeks_i)
                 -------------------------------------------------------------
                 (PRIOR_STRENGTH + Σ decay^age_weeks_i)

then floored at FLOOR so nothing is permanently frozen out.

- Recent promotions dominate (decay ≈ 0.9 per week).
- Unreviewed seeds (promoted == null) count as weak implicit negatives once they
  are UNREVIEWED_NEGATIVE_DAYS old — they didn't earn a look. Younger unreviewed
  seeds contribute no signal yet.
- The smoothing prior means an unseen tag scores PRIOR_MEAN (neutral), so
  exploration is preserved and small samples aren't over-trusted.

Run standalone (``python runner/weights.py``) to force a re-derivation, or let
``generate.py`` call ``maybe_refresh`` at the start of a run once 7 days pass.
"""

from __future__ import annotations

from datetime import date

import store

DECAY = 0.9                      # per-week decay: recent signal dominates
PRIOR_STRENGTH = 1.0             # pseudocount weight of the neutral prior
PRIOR_MEAN = 0.5                 # neutral promote-rate for an unseen/thin tag
FLOOR = 0.05                     # keep every tag reachable — no permanent freeze-out
UNREVIEWED_NEGATIVE_DAYS = 14    # unreviewed-and-old counts as a weak negative
REFRESH_INTERVAL_DAYS = 7        # re-derive at most weekly


def _tag_score(entries: list[dict], key: str, tag: str, today: date) -> float:
    """Score one tag from the log entries where entry[key] == tag."""
    numerator = PRIOR_STRENGTH * PRIOR_MEAN
    denominator = PRIOR_STRENGTH
    for entry in entries:
        if entry.get(key) != tag:
            continue
        age_days = (today - date.fromisoformat(entry["date"])).days
        promoted = entry.get("promoted")
        if promoted is True:
            value = 1.0
        elif promoted is False:
            value = 0.0
        elif promoted is None:
            if age_days < UNREVIEWED_NEGATIVE_DAYS:
                continue  # not a signal yet
            value = 0.0   # ignored long enough → weak implicit negative
        else:
            continue      # malformed promoted field — ignore defensively
        weight = DECAY ** (max(age_days, 0) / 7.0)
        numerator += value * weight
        denominator += weight
    return max(numerator / denominator, FLOOR)


def rederive(matrix: dict, entries: list[dict], today: date) -> dict:
    """Recompute tag scores for every domain and technique in the matrix."""
    return {
        "generated_at": today.isoformat(),
        "domains": {d: _tag_score(entries, "domain", d, today) for d in matrix["domains"]},
        "techniques": {
            t: _tag_score(entries, "technique", t, today) for t in matrix["techniques"]
        },
    }


def refresh_due(weights: dict, today: date) -> bool:
    generated_at = weights.get("generated_at")
    if not generated_at:
        return True
    return (today - date.fromisoformat(generated_at)).days >= REFRESH_INTERVAL_DAYS


def maybe_refresh(matrix: dict, weights: dict, entries: list[dict], today: date) -> dict:
    """Re-derive and persist weights if a week has passed; else return as-is."""
    if refresh_due(weights, today):
        weights = rederive(matrix, entries, today)
        store.save_json(store.WEIGHTS_PATH, weights)
    return weights


def main() -> None:
    store.configure_console()
    matrix = store.load_json(store.MATRIX_PATH)
    entries = store.read_log()
    today = store.today()
    weights = rederive(matrix, entries, today)
    store.save_json(store.WEIGHTS_PATH, weights)
    print(f"Re-derived weights for {today.isoformat()} from {len(entries)} log entries.")
    top_d = sorted(weights["domains"].items(), key=lambda kv: kv[1], reverse=True)[:3]
    top_t = sorted(weights["techniques"].items(), key=lambda kv: kv[1], reverse=True)[:3]
    print("  top domains:   ", ", ".join(f"{k}={v:.2f}" for k, v in top_d))
    print("  top techniques:", ", ".join(f"{k}={v:.2f}" for k, v in top_t))


if __name__ == "__main__":
    main()
