"""Shared paths and I/O for the seed farm.

Everything that touches disk lives here so ``generate.py``, ``weights.py``, and
``review.py`` all agree on where state lives and how it's read/written. Paths are
resolved relative to the repo root (the parent of this file's directory), so the
runner works regardless of the current working directory.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from datetime import date, datetime, timedelta, timezone
from pathlib import Path


def configure_console() -> None:
    """Make stdout/stderr tolerate any Unicode (e.g. a model title with an emoji)
    even on a legacy Windows code page, so a local run never dies on a print."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):
                pass

REPO_ROOT = Path(__file__).resolve().parent.parent
STATE_DIR = REPO_ROOT / "state"
SEEDS_DIR = REPO_ROOT / "seeds"
REVIEWS_DIR = REPO_ROOT / "reviews"

MATRIX_PATH = STATE_DIR / "matrix.json"
LOG_PATH = STATE_DIR / "log.jsonl"
WEIGHTS_PATH = STATE_DIR / "weights.json"
LAST_COMMIT_MSG_PATH = STATE_DIR / "last_commit_msg.txt"
LOG_MD_PATH = REPO_ROOT / "LOG.md"


# --------------------------------------------------------------------------- #
# Dates
# --------------------------------------------------------------------------- #
def today() -> date:
    """Today's date (UTC), overridable via SEED_MINER_DATE for testing."""
    override = os.environ.get("SEED_MINER_DATE")
    if override:
        return date.fromisoformat(override)
    return datetime.now(timezone.utc).date()


# --------------------------------------------------------------------------- #
# JSON helpers (atomic writes so a crash mid-write can't corrupt state)
# --------------------------------------------------------------------------- #
def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def save_json(path: Path, data: dict) -> None:
    _atomic_write(path, json.dumps(data, indent=2, ensure_ascii=False) + "\n")


def write_text(path: Path, text: str) -> None:
    _atomic_write(path, text)


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(text)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


# --------------------------------------------------------------------------- #
# The log (JSON Lines — one seed per line, append-only source of truth)
# --------------------------------------------------------------------------- #
def read_log() -> list[dict]:
    if not LOG_PATH.exists():
        return []
    entries = []
    with LOG_PATH.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def append_log(entry: dict) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(entry, ensure_ascii=False) + "\n")


def rewrite_log(entries: list[dict]) -> None:
    """Rewrite the whole log (used by review.py when editing an entry)."""
    body = "".join(json.dumps(e, ensure_ascii=False) + "\n" for e in entries)
    _atomic_write(LOG_PATH, body)


# --------------------------------------------------------------------------- #
# LOG.md — the human review surface, regenerated from the log every run
# --------------------------------------------------------------------------- #
def slug_of(entry: dict) -> str:
    """Display slug: the id with the leading ``YYYY-MM-DD_`` date stripped."""
    seed_id = entry.get("id", "")
    return seed_id.split("_", 1)[1] if "_" in seed_id else seed_id


def find_entry(entries: list[dict], ident: str) -> dict:
    """Return the single log entry matching a full identifier or a bare slug.

    Raises LookupError if there is no match, or more than one.
    """
    matches = [e for e in entries if e.get("id") == ident or slug_of(e) == ident]
    if not matches:
        raise LookupError(f"No seed matching {ident!r}.")
    if len(matches) > 1:
        joined = ", ".join(e["id"] for e in matches)
        raise LookupError(f"{ident!r} is ambiguous; use the full identifier: {joined}")
    return matches[0]


def week_of(iso_date: str) -> date:
    """Return the Monday of the week containing ``iso_date``."""
    d = date.fromisoformat(iso_date)
    return d - timedelta(days=d.weekday())


def render_log_md(entries: list[dict]) -> str:
    header = (
        "<!-- AUTO-GENERATED from state/log.jsonl on every run. Do NOT edit by hand. -->\n"
        "<!-- To promote/reject a seed:  "
        "python runner/review.py promote <id> --note \"why\" -->\n\n"
        "# Review log\n\n"
        "Checking a box = promoted. `[x]` promoted · `[~]` rejected · `[ ]` unreviewed. "
        "Promote with `python runner/review.py promote <id>`.\n"
    )
    if not entries:
        return header + (
            "\nNo seeds generated yet. Run `python runner/generate.py --dry-run` "
            "to try the pipeline locally.\n"
        )

    # Group by the Monday of each seed's week, newest week first.
    weeks: dict[date, list[dict]] = {}
    for entry in entries:
        weeks.setdefault(week_of(entry["date"]), []).append(entry)

    lines = [header]
    for monday in sorted(weeks, reverse=True):
        lines.append(f"\n## Week of {monday.isoformat()}\n")
        for entry in sorted(weeks[monday], key=lambda e: e["date"]):
            promoted = entry.get("promoted")
            box = "x" if promoted is True else "~" if promoted is False else " "
            title = entry.get("title", "(untitled)")
            lines.append(f"- [{box}] `{entry['id']}` — {title}")
            self_note = entry.get("self_assessment")
            if self_note:
                lines.append(f'      self: "{self_note}"')
            review_note = entry.get("review_note")
            if review_note:
                lines.append(f'      note: "{review_note}"')
            graduated = entry.get("graduated_to")
            if graduated:
                lines.append(f"      graduated: {graduated}")
    return "\n".join(lines).rstrip() + "\n"


def regenerate_log_md(entries: list[dict] | None = None) -> None:
    if entries is None:
        entries = read_log()
    write_text(LOG_MD_PATH, render_log_md(entries))
