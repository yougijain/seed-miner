"""Human review CLI — the one required action for the whole thing to work.

    python runner/review.py list                       # show seeds + review status
    python runner/review.py promote <id> --note "why"  # checking the box
    python runner/review.py reject  <id> --note "why"

Promoting/rejecting edits the seed's line in state/log.jsonl, regenerates LOG.md,
and re-derives state/weights.json so the next run immediately leans toward what
you liked. <id> may be the full id (2026-07-17_disc-golf-network) or just the
slug (disc-golf-network) when unambiguous.

Promotion is a signal, not the work: the payoff is forking a promoted seed into
its own real repo and diverging from it by hand.
"""

from __future__ import annotations

import argparse
import sys

import store
import weights as weights_mod


def _find(entries: list[dict], ident: str) -> dict:
    matches = [e for e in entries if e["id"] == ident or store.slug_of(e) == ident]
    if not matches:
        sys.exit(f"No seed matching '{ident}'. Try `review.py list`.")
    if len(matches) > 1:
        ids = ", ".join(e["id"] for e in matches)
        sys.exit(f"'{ident}' is ambiguous — use the full id. Candidates: {ids}")
    return matches[0]


def _set_status(ident: str, promoted: bool, note: str | None) -> None:
    entries = store.read_log()
    entry = _find(entries, ident)
    entry["promoted"] = promoted
    if note is not None:
        entry["review_note"] = note
    store.rewrite_log(entries)
    store.regenerate_log_md(entries)
    # Re-derive weights now so the promotion steers the very next run.
    matrix = store.load_json(store.MATRIX_PATH)
    weights = weights_mod.rederive(matrix, entries, store.today())
    store.save_json(store.WEIGHTS_PATH, weights)
    verb = "Promoted" if promoted else "Rejected"
    print(f"{verb} {entry['id']}. LOG.md and weights updated.")


def _list() -> None:
    entries = store.read_log()
    if not entries:
        print("No seeds yet.")
        return
    for e in entries:
        promoted = e.get("promoted")
        mark = "[x]" if promoted is True else "[~]" if promoted is False else "[ ]"
        print(f"{mark} {e['id']}  —  {e.get('title', '')}")
        if e.get("self_assessment"):
            print(f'      self: "{e["self_assessment"]}"')


def main() -> None:
    parser = argparse.ArgumentParser(description="Review (promote/reject) generated seeds.")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("list", help="List all seeds and their review status.")
    for name in ("promote", "reject"):
        p = sub.add_parser(name, help=f"Mark a seed as {name}d.")
        p.add_argument("id", help="Seed id or slug.")
        p.add_argument("--note", default=None, help="Optional one-line reason (recorded).")

    args = parser.parse_args()
    if args.command == "list":
        _list()
    elif args.command == "promote":
        _set_status(args.id, True, args.note)
    elif args.command == "reject":
        _set_status(args.id, False, args.note)


if __name__ == "__main__":
    main()
