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
from pathlib import Path

import digest
import prompt as prompt_mod
import store
import weights as weights_mod

# ---- Model and sampling configuration -------------------------------------- #
DEFAULT_MODEL = "claude-haiku-4-5"     # override with SEED_MINER_MODEL
DEFAULT_MAX_TOKENS = 8000              # upper bound on output tokens per run

NOVELTY_BONUS = 1.5                    # multiplier for a tag absent from recent runs
OBVIOUS_PENALTY = 0.3                  # multiplier for a domain's obvious technique
NOVELTY_WINDOW = 10                    # runs considered "recent" for the novelty bonus
REPEAT_WINDOW = 30                     # seeds shown to the model to discourage repeats

REQUIRED_KEYS = ("title", "slug", "one_line", "files")


# --------------------------------------------------------------------------- #
# Cell sampling
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


def _flatten(text: str, limit: int = 300) -> str:
    """Collapse whitespace and truncate a field taken from a previous seed.

    Earlier titles are fed back into the next prompt. Collapsing newlines keeps
    each entry on a single list item, so text authored by a previous generation
    cannot introduce new structure into the prompt.
    """
    return " ".join(str(text).split())[:limit]


def recent_digest(entries: list[dict]) -> str:
    """Build the recent-seeds list used by the prompt's avoid-repeats section."""
    lines = [
        f"- {_flatten(e.get('title', '(untitled)'))}: {_flatten(e.get('one_line', ''))}".rstrip()
        for e in entries[-REPEAT_WINDOW:]
    ]
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
    _validate(data)
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


def _validate(data: object) -> None:
    """Reject malformed model output before anything is written to disk."""
    if not isinstance(data, dict):
        raise SystemExit("Model output is not a JSON object — committing nothing.")
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
    """Write the generated files into ``seed_dir``.

    Every destination is validated before anything is written, so a seed with an
    invalid path is rejected without leaving a partial directory behind.
    """
    seed_root = seed_dir.resolve()
    planned: list[tuple[Path, str]] = []
    for f in files:
        # Normalise separators so a path behaves identically on Windows and Linux
        # (a backslash is a literal filename character on POSIX), then drop any
        # leading separator so an absolute path cannot escape the seed directory.
        rel = str(f["path"]).replace("\\", "/").strip().lstrip("/")
        if not rel or rel in (".", ".."):
            raise SystemExit(f"Invalid file path in model output: {f['path']!r}")
        target = (seed_dir / rel).resolve()
        # Confine writes to the seed directory; reject any traversal attempt.
        if seed_root != target and seed_root not in target.parents:
            raise SystemExit(f"Refusing to write outside the seed directory: {f['path']!r}")
        planned.append((target, str(f["content"])))
    for target, content in planned:
        target.parent.mkdir(parents=True, exist_ok=True)
        store.write_text(target, content)


# --------------------------------------------------------------------------- #
# Dry-run stub
#
# Produces a minimal but valid seed so the sample -> write -> log -> review
# pipeline can be exercised locally without an API key or any cost.
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
        "Automatically generated seed (dry-run stub). Produced by the local "
        "`--dry-run` path rather than the model; it exists only to verify that the "
        "generation pipeline works end to end.\n\n"
        f"**Question:** does a two-sigma rule flag the planted spike in a synthetic "
        f"*{domain}* series?\n\n"
        "**Data:** fully synthetic.\n\n"
        "**Limitation:** this is a stub. It contains no non-obvious adaptation of "
        "the technique and is not intended for review.\n"
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
    digest.regenerate(today)

    # Strip first, then fall back — a whitespace-only value would otherwise pass
    # the `or` check and produce an empty `git commit -m`, which git rejects.
    commit_msg = str(data.get("commit_message") or "").strip()
    if not commit_msg:
        commit_msg = f"feat(seed): {domain} x {technique} — {slug}"
    store.write_text(store.LAST_COMMIT_MSG_PATH, commit_msg + "\n")

    print(f"Wrote seed: seeds/{seed_id}/  ({len(data['files'])} files)")
    print(f"Commit message: {commit_msg}")


if __name__ == "__main__":
    main()
