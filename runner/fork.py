"""Graduate a promoted seed into its own repository.

Copies a seed out of the farm into a new standalone repository, adds the
scaffolding a seed lacks (project README, requirements, .gitignore), makes the
initial commit, pushes, and records the resulting URL back in the seed log.

    python runner/fork.py <id> [--name NAME] [--private] [--dry-run]

The initial commit is attributed to the local git identity, with the seed's
origin recorded in the commit message. Everything after that commit is ordinary
human work on an ordinary repository.

Requires the GitHub CLI (``gh``) to be installed and authenticated. No token is
stored anywhere; this is a local command run deliberately, not automation.
"""

from __future__ import annotations

import argparse
import ast
import shutil
import subprocess
import sys
import tempfile
from datetime import date
from pathlib import Path

import digest
import store

# Import names that differ from their distribution name on PyPI.
PACKAGE_ALIASES = {
    "sklearn": "scikit-learn",
    "cv2": "opencv-python",
    "PIL": "pillow",
    "yaml": "PyYAML",
    "bs4": "beautifulsoup4",
    "dateutil": "python-dateutil",
    "mpl_toolkits": "matplotlib",
}

GITIGNORE = """\
__pycache__/
*.py[cod]
.venv/
venv/
.env
.ipynb_checkpoints/
.DS_Store
"""


def _run(cmd: list[str], cwd: Path | None = None) -> subprocess.CompletedProcess:
    """Run a command from an argument list (never a shell string).

    Failures are reported as a clean message rather than a traceback.
    """
    try:
        return subprocess.run(cmd, cwd=cwd, check=True, capture_output=True, text=True)
    except FileNotFoundError:
        raise SystemExit(f"Required command not found: {cmd[0]}")
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or exc.stdout or "").strip()
        raise SystemExit(f"Command failed: {' '.join(cmd)}\n{detail}")


def farm_repo() -> str | None:
    """Return ``owner/name`` for the farm, parsed from the git remote."""
    try:
        proc = subprocess.run(
            ["git", "remote", "get-url", "origin"],
            cwd=store.REPO_ROOT, capture_output=True, text=True,
        )
    except FileNotFoundError:
        return None
    if proc.returncode != 0:
        return None
    url = proc.stdout.strip().removesuffix(".git")
    if url.startswith("git@"):           # git@github.com:owner/name
        _, _, path = url.partition(":")
        return path or None
    parts = url.rstrip("/").split("/")   # https://github.com/owner/name
    return "/".join(parts[-2:]) if len(parts) >= 2 else None


def infer_requirements(py_files: list[Path]) -> list[str]:
    """Third-party distributions imported by the seed's Python files."""
    modules: set[str] = set()
    for path in py_files:
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeDecodeError):
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules.update(a.name.split(".")[0] for a in node.names)
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                modules.add(node.module.split(".")[0])
    third_party = {m for m in modules if m and m not in sys.stdlib_module_names}
    return sorted({PACKAGE_ALIASES.get(m, m) for m in third_party})


def repo_name_for(entry: dict, override: str | None) -> str:
    raw = override or store.slug_of(entry)
    name = "".join(c if (c.isalnum() or c in "-_.") else "-" for c in raw).strip("-.")
    if not name:
        raise SystemExit("Could not derive a valid repository name; pass --name.")
    return name


def build_readme(entry: dict, seed_url: str | None, has_seed_doc: bool) -> str:
    title = entry.get("title", "Untitled project")
    lines = [f"# {title}", ""]
    if entry.get("one_line"):
        lines += [entry["one_line"], ""]

    lines += [
        "## Status",
        "",
        "Early stage. This repository began as a generated project seed and is",
        "being developed further by hand.",
        "",
        "## Running",
        "",
        "```bash",
        "pip install -r requirements.txt",
        "python main.py",
        "```",
        "",
        "## Origin",
        "",
    ]

    origin = (
        f"This project started as seed `{entry.get('id', '?')}`, generated on "
        f"{entry.get('date', '?')} for the `{entry.get('domain', '?')}` x "
        f"`{entry.get('technique', '?')}` pairing"
    )
    origin += f" by [seed-miner]({seed_url})." if seed_url else " by seed-miner."
    lines += [origin, ""]

    if entry.get("self_assessment"):
        lines += [f"Assessment recorded at generation time: {entry['self_assessment']}", ""]
    if entry.get("review_note"):
        lines += [f"Review note: {entry['review_note']}", ""]
    if has_seed_doc:
        lines += ["The seed's original documentation is preserved in [SEED.md](SEED.md).", ""]

    lines += [
        "The initial commit contains the generated seed unchanged. Every commit",
        "after it is original work.",
    ]
    return "\n".join(lines) + "\n"


def stage_repo(entry: dict, staging: Path, seed_url: str | None) -> list[str]:
    """Populate ``staging`` with the new repository's contents."""
    seed_id = entry["id"]
    seed_dir = store.SEEDS_DIR / seed_id
    if not seed_dir.is_dir():
        raise SystemExit(f"Seed directory not found: seeds/{seed_id}")

    created: list[str] = []
    seed_doc = None
    for src in sorted(p for p in seed_dir.rglob("*") if p.is_file()):
        rel = src.relative_to(seed_dir)
        # Preserve the seed's own README separately; the project gets a new one.
        if rel.as_posix().lower() == "readme.md":
            seed_doc = src
            continue
        dest = staging / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
        created.append(rel.as_posix())

    if seed_doc is not None:
        shutil.copy2(seed_doc, staging / "SEED.md")
        created.append("SEED.md")

    (staging / "README.md").write_text(
        build_readme(entry, seed_url, seed_doc is not None), encoding="utf-8", newline="\n"
    )
    created.append("README.md")

    (staging / ".gitignore").write_text(GITIGNORE, encoding="utf-8", newline="\n")
    created.append(".gitignore")

    requirements = infer_requirements([p for p in staging.rglob("*.py") if p.is_file()])
    if requirements:
        (staging / "requirements.txt").write_text(
            "\n".join(requirements) + "\n", encoding="utf-8", newline="\n"
        )
        created.append(f"requirements.txt ({', '.join(requirements)})")
    return created


def graduate(ident: str, name: str | None, private: bool, force: bool, dry_run: bool) -> None:
    entries = store.read_log()
    try:
        entry = store.find_entry(entries, ident)
    except LookupError as exc:
        sys.exit(str(exc))

    if entry.get("promoted") is not True and not force:
        sys.exit(
            f"Seed {entry['id']} is not promoted (promoted={entry.get('promoted')!r}). "
            "Promote it first, or pass --force."
        )
    if entry.get("graduated_to"):
        sys.exit(f"Seed {entry['id']} was already graduated to {entry['graduated_to']}.")

    farm = farm_repo()
    seed_url = f"https://github.com/{farm}/tree/main/seeds/{entry['id']}" if farm else None
    repo_name = repo_name_for(entry, name)
    visibility = "--private" if private else "--public"

    print(f"Seed:       {entry.get('title', '?')}")
    print(f"Identifier: {entry['id']}")
    print(f"Repository: {repo_name} ({visibility.lstrip('-')})")

    with tempfile.TemporaryDirectory() as tmp:
        staging = Path(tmp)
        created = stage_repo(entry, staging, seed_url)
        for item in created:
            print(f"  {item}")

        if dry_run:
            print("\n--- README.md preview ---")
            print((staging / "README.md").read_text(encoding="utf-8"))
            print("Dry run: no repository created, nothing pushed, log unchanged.")
            return

        message = (
            f"Import seed {entry['id']} from seed-miner\n\n"
            f"Generated {entry.get('date', '?')} for the "
            f"{entry.get('domain', '?')} x {entry.get('technique', '?')} pairing."
        )
        if seed_url:
            message += f"\nSource: {seed_url}"

        # Initial commit uses the local git identity; provenance lives in the message.
        _run(["git", "init", "-q", "-b", "main"], cwd=staging)
        _run(["git", "add", "-A"], cwd=staging)
        _run(["git", "commit", "-q", "-m", message], cwd=staging)

        create = ["gh", "repo", "create", repo_name, visibility, "--source=.", "--push"]
        if entry.get("one_line"):
            create += ["--description", entry["one_line"][:340]]
        result = _run(create, cwd=staging)

        owner = farm.split("/")[0] if farm else None
        fallback = f"https://github.com/{owner}/{repo_name}" if owner else repo_name
        url = next(
            (ln.strip() for ln in result.stdout.splitlines() if ln.strip().startswith("http")),
            fallback,
        )

    entry["graduated_to"] = url
    entry["graduated_at"] = store.today().isoformat()
    store.rewrite_log(entries)
    store.regenerate_log_md(entries)
    digest.regenerate(date.fromisoformat(entry["date"]), entries)

    print(f"\nPushed to {url}")
    print("Recorded graduation in state/log.jsonl, LOG.md, and the week's digest.")


def main() -> None:
    store.configure_console()
    parser = argparse.ArgumentParser(
        description="Graduate a promoted seed into its own repository."
    )
    parser.add_argument("id", help="Seed identifier or slug.")
    parser.add_argument("--name", help="Repository name. Defaults to the seed's slug.")
    parser.add_argument(
        "--private", action="store_true", help="Create a private repository (default: public)."
    )
    parser.add_argument(
        "--force", action="store_true", help="Graduate even if the seed is not promoted."
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show what would be created without creating or pushing anything.",
    )
    args = parser.parse_args()
    graduate(args.id, args.name, args.private, args.force, args.dry_run)


if __name__ == "__main__":
    main()
