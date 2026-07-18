"""The runtime generation prompt (spec Section 5).

``generate.py`` samples a {domain, technique} cell, fetches recent seeds to avoid
repeats, and injects both into ``GENERATION_PROMPT`` below. We substitute the
three placeholders with str.replace (not str.format / f-strings) so the literal
``{ ... }`` JSON braces inside the prompt are left untouched.
"""

from __future__ import annotations

# A short system prompt reinforces the strict-JSON contract for an unattended run.
SYSTEM_PROMPT = (
    "You are a precise generator of small data-analyst project seeds. "
    "You output exactly one JSON object and nothing else: no markdown fences, "
    "no preamble, no trailing commentary."
)

GENERATION_PROMPT = """You are generating ONE small, self-contained data-analyst project seed for a public "seed farm" repo. This is exploratory scratch work — most seeds will be discarded. Your job is to make THIS cell of the domain×technique matrix produce something a curious person wouldn't have deliberately picked, but that has a real chance of being interesting.

CELL FOR THIS RUN:
- Domain: {domain}
- Technique: {technique}
- Target obviousness: NON-OBVIOUS. Do not produce the safe, tutorial-grade pairing. Find the angle where applying {technique} to {domain} forces a genuine modification because the technique wasn't built for this domain's data shape. The seams are the point.

AVOID REPEATING (last ~30 seeds — do not overlap in concept, dataset, or framing):
{recent_titles_and_onelines}

HARD CONSTRAINTS:
- Total code under 200 lines across all files.
- Runs standalone. stdlib + at most one common package (pandas / numpy / networkx / scikit-learn / plotly — pick what fits). No API keys, no paid data.
- If real public data isn't plausibly available for this domain, generate a small synthetic dataset inline that has the RIGHT STRUCTURE to make the technique meaningful (i.e. the anomaly/pattern/signal actually exists in the data — don't fake a result on noise).
- The project must answer ONE narrow, concrete question. State it in the README.
- Be honest in the README about what's real vs synthetic and what the limitation is.

QUALITY BAR — the seed is worth generating only if a reviewer could look at it and say "huh, that modification to {technique} is actually the interesting bit." If the honest answer is "this is just {technique} on a toy dataset," say so in a `"self_assessment"` field rather than dressing it up.

OUTPUT — strict JSON, no markdown fences, no preamble:
{
  "title": "short descriptive title",
  "slug": "kebab-case-slug",
  "one_line": "one sentence: what question it answers + the non-obvious angle",
  "obviousness": "obvious | non_obvious",
  "self_assessment": "one honest sentence on whether this has legs or is filler",
  "commit_message": "conventional-commit style",
  "files": [
    { "path": "main.py", "content": "..." },
    { "path": "README.md", "content": "states the question, real-vs-synthetic, the limitation, and that this is an auto-generated seed" }
  ]
}
Return only the JSON object."""


def build_prompt(domain: str, technique: str, recent: str) -> str:
    """Inject the sampled cell and recent-seed digest into the prompt template."""
    return (
        GENERATION_PROMPT.replace("{domain}", domain)
        .replace("{technique}", technique)
        .replace("{recent_titles_and_onelines}", recent or "(none yet — this is an early run)")
    )
