"""Weekly review digests.

Writes one Markdown file per week under ``reviews/`` summarising every seed
produced that week: the cell it came from, the question it answers, the model's
own assessment, its current review status, and links to its files. The digest is
the surface used to decide which seeds are worth developing further.

Digests are regenerated automatically by ``generate.py`` (for the current week)
and by ``review.py`` (for the week a decision affects). To rebuild them by hand:

    python runner/digest.py                    # rebuild every week
    python runner/digest.py --week 2026-07-20  # rebuild one week (any date in it)
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import date, timedelta

import store

STATUS_LABELS = {True: "Promoted", False: "Rejected", None: "Unreviewed"}


def _status(entry: dict) -> str:
    return STATUS_LABELS.get(entry.get("promoted"), "Unreviewed")


def digest_path(monday: date):
    return store.REVIEWS_DIR / f"week-of-{monday.isoformat()}.md"


def _file_links(seed_id: str) -> str:
    """Markdown links from reviews/ to each file of a seed."""
    seed_dir = store.SEEDS_DIR / seed_id
    if not seed_dir.is_dir():
        return "_files not found_"
    files = sorted(p for p in seed_dir.rglob("*") if p.is_file())
    if not files:
        return "_no files_"
    return " · ".join(
        f"[{p.relative_to(seed_dir).as_posix()}]"
        f"(../seeds/{seed_id}/{p.relative_to(seed_dir).as_posix()})"
        for p in files
    )


def render_week(monday: date, entries: list[dict]) -> str:
    sunday = monday + timedelta(days=6)
    counts = Counter(_status(e) for e in entries)
    lines = [
        "<!-- AUTO-GENERATED from state/log.jsonl. Do not edit by hand. -->",
        "",
        f"# Seeds — week of {monday.isoformat()}",
        "",
        f"Covering {monday.isoformat()} to {sunday.isoformat()}.",
        "",
        f"{len(entries)} generated · {counts['Promoted']} promoted · "
        f"{counts['Rejected']} rejected · {counts['Unreviewed']} unreviewed",
        "",
        "Record a decision with:",
        "",
        "```bash",
        'python runner/review.py promote <id> --note "reason"',
        'python runner/review.py reject  <id> --note "reason"',
        "```",
    ]

    if not entries:
        lines += ["", "No seeds were generated this week."]
        return "\n".join(lines) + "\n"

    for n, entry in enumerate(sorted(entries, key=lambda e: e["date"]), start=1):
        seed_id = entry.get("id", "")
        lines += [
            "",
            "---",
            "",
            f"## {n}. {entry.get('title', '(untitled)')}",
            "",
            f"`{entry.get('domain', '?')}` × `{entry.get('technique', '?')}` · "
            f"{entry.get('obviousness', '?')} · generated {entry.get('date', '?')} · "
            f"**{_status(entry)}**",
            "",
            f"Identifier: `{seed_id}`",
        ]
        if entry.get("one_line"):
            lines += ["", f"**Question.** {entry['one_line']}"]
        if entry.get("self_assessment"):
            lines += ["", f"**Model's assessment.** {entry['self_assessment']}"]
        if entry.get("review_note"):
            lines += ["", f"**Review note.** {entry['review_note']}"]
        lines += ["", f"**Files.** {_file_links(seed_id)}"]

    return "\n".join(lines) + "\n"


def render_index(weeks: dict[date, list[dict]]) -> str:
    lines = [
        "<!-- AUTO-GENERATED from state/log.jsonl. Do not edit by hand. -->",
        "",
        "# Weekly review digests",
        "",
        "One file per week summarising the seeds generated in that week.",
        "",
        "| Week | Generated | Promoted | Rejected | Unreviewed |",
        "|------|-----------|----------|----------|------------|",
    ]
    for monday in sorted(weeks, reverse=True):
        entries = weeks[monday]
        c = Counter(_status(e) for e in entries)
        lines.append(
            f"| [Week of {monday.isoformat()}](week-of-{monday.isoformat()}.md) "
            f"| {len(entries)} | {c['Promoted']} | {c['Rejected']} | {c['Unreviewed']} |"
        )
    if not weeks:
        lines.append("| _none yet_ | 0 | 0 | 0 | 0 |")
    return "\n".join(lines) + "\n"


def _weeks(entries: list[dict]) -> dict[date, list[dict]]:
    weeks: dict[date, list[dict]] = {}
    for entry in entries:
        weeks.setdefault(store.week_of(entry["date"]), []).append(entry)
    return weeks


def regenerate(target: date | None = None, entries: list[dict] | None = None) -> None:
    """Rebuild the digest for ``target``'s week, or every week when target is None.

    The index is always rebuilt so week counts stay accurate.
    """
    if entries is None:
        entries = store.read_log()
    weeks = _weeks(entries)

    if target is None:
        for monday, week_entries in weeks.items():
            store.write_text(digest_path(monday), render_week(monday, week_entries))
    else:
        monday = store.week_of(target.isoformat())
        store.write_text(digest_path(monday), render_week(monday, weeks.get(monday, [])))

    store.write_text(store.REVIEWS_DIR / "README.md", render_index(weeks))


def main() -> None:
    store.configure_console()
    parser = argparse.ArgumentParser(description="Rebuild weekly review digests.")
    parser.add_argument(
        "--week",
        metavar="YYYY-MM-DD",
        help="Rebuild only the week containing this date. Defaults to all weeks.",
    )
    args = parser.parse_args()

    target = date.fromisoformat(args.week) if args.week else None
    regenerate(target)
    written = "one week" if target else "all weeks"
    print(f"Rebuilt review digests ({written}) under {store.REVIEWS_DIR.name}/.")


if __name__ == "__main__":
    main()
