"""Main loop: sample a cell → call Claude → write the seed → log it.

Run:
    python runner/generate.py            # real run, needs ANTHROPIC_API_KEY
    python runner/generate.py --dry-run  # stubbed seed, no API call, no cost

Environment overrides (all optional):
    SEED_MINER_MODEL       default "claude-haiku-4-5"
    SEED_MINER_MAX_TOKENS  default 8000  (hard ceiling so a run can't balloon)
    SEED_MINER_DATE        override "today" as YYYY-MM-DD (testing)
    SEED_MINER_SEED        seed the RNG for reproducible cell sampling (testing)
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from pathlib import Path

import prompt as prompt_mod
import store
import weights as weights_mod

# ---- Model / sampling configuration ---------------------------------------- #
DEFAULT_MODEL = "claude-haiku-4-5"     # spec default; Sonnet if you want less-generic code
DEFAULT_MAX_TOKENS = 8000              # hard ceiling — a runaway generation can't balloon

NOVELTY_BONUS = 1.5                    # reward tags absent from the last N runs
OBVIOUS_PENALTY = 0.3                  # down-weight the hardcoded obvious pairing per domain
NOVELTY_WINDOW = 10                    # "recent" window for the novelty bonus
REPEAT_WINDOW = 30                     # how many recent seeds to show the model to avoid repeats

REQUIRED_KEYS = ("title", "slug", "one_line", "files")


# --------------------------------------------------------------------------- #
# Cell sampling (spec Section 6, per-run part)
# --------------------------------------------------------------------------- #
def choose_cell(matrix: dict, weights: dict, entries: list[dict], rng: random.Random):
    """Sample one {domain, technique} cell proportional to its score."""
    skip = {tuple(pair) for pair in matrix.get("skip", [])}
    obvious = matrix.get("obvious_pairings", {})
    recent = entries[-NOVELTY_WINDOW:]
    recent_domains = {e["domain"] for e in recent}
    recent_techniques = {e["technique"] for e in recent}

    cells, scores = [], []
    for domain in matrix["domains"]:
        for technique in matrix["techniques"]:
            if (domain, technique) in skip:
                continue
            w_dom = weights["domains"].get(domain, weights_mod.FLOOR)
            w_tech = weights["techniques"].get(technique, weights_mod.FLOOR)
            novelty = 1.0
            if domain not in recent_domains:
                novelty *= NOVELTY_BONUS
            if technique not in recent_techniques:
                novelty *= NOVELTY_BONUS
            non_obvious = OBVIOUS_PENALTY if obvious.get(domain) == technique else 1.0
            cells.append((domain, technique))
            scores.append(w_dom * w_tech * novelty * non_obvious)

    if not cells:
        raise SystemExit("No valid matrix cells to sample — check state/matrix.json.")
    if sum(scores) <= 0:  # degenerate: fall back to uniform
        scores = [1.0] * len(cells)
    return rng.choices(cells, weights=scores, k=1)[0]


def recent_digest(entries: list[dict]) -> str:
    """A compact list of recent titles + one-liners for the AVOID-REPEATS block."""
    lines = []
    for e in entries[-REPEAT_WINDOW:]:
        lines.append(f"- {e.get('title', '(untitled)')}: {e.get('one_line', '')}".rstrip())
    return "\n".join(reversed(lines))  # most recent first


# --------------------------------------------------------------------------- #
# Generation
# --------------------------------------------------------------------------- #
def generate_seed(domain: str, technique: str, recent: str, dry_run: bool) -> dict:
    if dry_run:
        return _stub_seed(domain, technique)

    from anthropic import Anthropic  # imported lazily so --dry-run needs no SDK

    model = os.environ.get("SEED_MINER_MODEL", DEFAULT_MODEL)
    max_tokens = int(os.environ.get("SEED_MINER_MAX_TOKENS", DEFAULT_MAX_TOKENS))
    client = Anthropic()  # reads ANTHROPIC_API_KEY from the environment
    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=prompt_mod.SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt_mod.build_prompt(domain, technique, recent)}],
    )
    if response.stop_reason == "max_tokens":
        raise SystemExit(
            f"Model hit the max_tokens ceiling ({max_tokens}) — output truncated, "
            "refusing to write a partial seed."
        )
    text = "".join(block.text for block in response.content if block.type == "text")
    data = _parse_json(text)
    _validate(data, domain, technique)
    return data


def _parse_json(text: str) -> dict:
    """Parse the model's JSON, defensively stripping fences / preamble if present."""
    stripped = text.strip()
    if stripped.startswith("```"):  # tolerate an accidental ```json ... ``` fence
        stripped = re.sub(r"^```[a-zA-Z]*\n?", "", stripped)
        stripped = re.sub(r"\n?```$", "", stripped).strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass
    # Last resort: grab the outermost {...} span and try again.
    start, end = stripped.find("{"), stripped.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(stripped[start : end + 1])
        except json.JSONDecodeError:
            pass
    raise SystemExit("Could not parse JSON from the model response — committing nothing.")


def _validate(data: dict, domain: str, technique: str) -> None:
    for key in REQUIRED_KEYS:
        if key not in data:
            raise SystemExit(f"Model output missing required key '{key}' — committing nothing.")
    files = data["files"]
    if not isinstance(files, list) or not files:
        raise SystemExit("Model output has no files — committing nothing.")
    for f in files:
        if not isinstance(f, dict) or "path" not in f or "content" not in f:
            raise SystemExit("A file entry is missing 'path'/'content' — committing nothing.")


# --------------------------------------------------------------------------- #
# Writing the seed to disk
# --------------------------------------------------------------------------- #
def slugify(raw: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", str(raw).lower()).strip("-")
    return s or "untitled"


def unique_seed_id(base_id: str) -> str:
    """Ensure the seed folder name is unique (rare same-day, same-slug collision)."""
    candidate, n = base_id, 2
    while (store.SEEDS_DIR / candidate).exists():
        candidate = f"{base_id}-{n}"
        n += 1
    return candidate


def write_seed_files(seed_dir: Path, files: list[dict]) -> None:
    seed_root = seed_dir.resolve()
    for f in files:
        rel = str(f["path"]).lstrip("/\\")
        target = (seed_dir / rel).resolve()
        # Reject path traversal — a generated 'path' must stay inside the seed dir.
        if seed_root != target and seed_root not in target.parents:
            raise SystemExit(f"Refusing to write outside the seed dir: {f['path']}")
        target.parent.mkdir(parents=True, exist_ok=True)
        store.write_text(target, str(f["content"]))


# --------------------------------------------------------------------------- #
# Dry-run stub — a real, tiny, honest seed so the full pipeline is exercisable
# without an API key.
# --------------------------------------------------------------------------- #
def _stub_seed(domain: str, technique: str) -> dict:
    slug = f"{slugify(domain)}-{slugify(technique)}-stub"
    title = f"[stub] {domain} × {technique}"
    main_py = (
        '"""Dry-run stub seed — NOT model-generated.\n\n'
        f"Cell: {domain} x {technique}. Exists to exercise the pipeline end to end\n"
        'without calling the API. Replace by running generate.py with an API key.\n"""\n\n'
        "import statistics\n\n"
        "# Synthetic series with one planted anomaly so there is a real signal to find.\n"
        "series = [10, 11, 9, 10, 12, 11, 42, 10, 9, 11]\n"
        "mean = statistics.mean(series)\n"
        "stdev = statistics.pstdev(series)\n"
        "outliers = [i for i, x in enumerate(series) if abs(x - mean) > 2 * stdev]\n\n"
        'if __name__ == "__main__":\n'
        '    print(f"mean={mean:.2f} stdev={stdev:.2f} outlier_indices={outliers}")\n'
    )
    readme = (
        f"# {title}\n\n"
        "**Auto-generated seed (dry-run stub).** This one was produced by the local "
        "`--dry-run` path, not the model — it only demonstrates that the farm's "
        "sample → write → log → review pipeline works end to end.\n\n"
        f"**Question:** does a trivial 2-sigma rule flag the planted spike in a "
        f"synthetic *{domain}* series? (Yes — that's the point of the plant.)\n\n"
        "**Real vs synthetic:** fully synthetic. **Limitation:** it's a stub; there "
        "is no non-obvious modification of the technique here.\n"
    )
    return {
        "title": title,
        "slug": slug,
        "one_line": f"Stub demonstrating the pipeline for the {domain} x {technique} cell.",
        "obviousness": "obvious",
        "self_assessment": "dry-run stub — filler by construction, exists only to test the pipeline.",
        "commit_message": f"chore(seed): dry-run stub for {domain} x {technique}",
        "files": [
            {"path": "main.py", "content": main_py},
            {"path": "README.md", "content": readme},
        ],
    }


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def main() -> None:
    store.configure_console()
    parser = argparse.ArgumentParser(description="Generate one data-analyst project seed.")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Produce a stub seed without calling the API (no key, no cost).",
    )
    args = parser.parse_args()

    seed_env = os.environ.get("SEED_MINER_SEED")
    rng = random.Random(int(seed_env)) if seed_env else random.Random()

    today = store.today()
    matrix = store.load_json(store.MATRIX_PATH)
    weights = store.load_json(store.WEIGHTS_PATH)
    entries = store.read_log()

    # Weekly re-derivation (or start-of-run if 7 days have passed / weights never set).
    weights = weights_mod.maybe_refresh(matrix, weights, entries, today)

    domain, technique = choose_cell(matrix, weights, entries, rng)
    print(f"Sampled cell: {domain} x {technique}")

    data = generate_seed(domain, technique, recent_digest(entries), args.dry_run)

    slug = slugify(data.get("slug") or data["title"])
    seed_id = unique_seed_id(f"{today.isoformat()}_{slug}")
    seed_dir = store.SEEDS_DIR / seed_id
    write_seed_files(seed_dir, data["files"])

    obviousness = data.get("obviousness", "non_obvious")
    if obviousness not in ("obvious", "non_obvious"):
        obviousness = "non_obvious"

    entry = {
        "id": seed_id,
        "date": today.isoformat(),
        "domain": domain,
        "technique": technique,
        "obviousness": obviousness,
        "title": data["title"],
        "one_line": data.get("one_line", ""),
        "self_assessment": data.get("self_assessment", ""),
        "promoted": None,
        "review_note": None,
    }
    store.append_log(entry)
    store.regenerate_log_md()

    commit_msg = data.get("commit_message") or f"feat(seed): {domain} x {technique} — {slug}"
    store.write_text(store.LAST_COMMIT_MSG_PATH, commit_msg.strip() + "\n")

    print(f"Wrote seed: seeds/{seed_id}/  ({len(data['files'])} files)")
    print(f"Commit message: {commit_msg.strip()}")


if __name__ == "__main__":
    main()
