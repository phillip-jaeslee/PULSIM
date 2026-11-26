#!/usr/bin/env python3
"""check_reproducible.py -- does this repository work for someone who just cloned it?

The unit tests answer "is the physics right". This answers a different question:
"does the thing we tell people to run actually run", from a clean checkout, with
no files that only exist on a maintainer's laptop.

It executes
  * every notebook at the repository root, headlessly, with the setup cell that
    clones and pip-installs PULSIM removed -- in CI the package is already
    installed, and re-cloning into the clone is not what a Colab user does;
  * every script in tutorials/ that does not read from wave/.

Tutorials that DO read from wave/ are reported by name, loudly, as unverified.
wave/ is gitignored pending a licence review, so those tutorials cannot run for
anyone outside the group; that is a real gap in the repository, not a detail to
hide behind a silent skip. They do not fail the run -- the exit status is about
what we ship, and we do not ship wave/.

Usage:
    python tools/check_reproducible.py            # everything
    python tools/check_reproducible.py --notebooks-only
    python tools/check_reproducible.py --tutorials-only
    python tools/check_reproducible.py --list     # classify, run nothing
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
WAVE_MARKER = "wave/"
CELL_TIMEOUT = 900          # seconds, per notebook cell
TUTORIAL_TIMEOUT = 900      # seconds, per tutorial


# ----------------------------------------------------------------- discovery
def notebooks() -> list[Path]:
    return sorted(p for p in REPO.glob("*.ipynb") if not p.name.startswith("."))


def tutorials() -> tuple[list[Path], list[Path]]:
    """(runnable, wave_dependent) -- split by whether the source reads wave/."""
    runnable, wave_dependent = [], []
    for path in sorted((REPO / "tutorials").glob("*.py")):
        text = path.read_text(encoding="utf-8", errors="replace")
        (wave_dependent if WAVE_MARKER in text else runnable).append(path)
    return runnable, wave_dependent


def is_setup_cell(source: str) -> bool:
    """The clone-and-install cell, which CI must not run: the package is already
    installed, and `git clone` into a non-empty directory fails anyway."""
    return "git clone" in source or "pip install" in source


# ----------------------------------------------------------------- execution
def run_notebook(path: Path) -> tuple[bool, str, float]:
    import nbformat
    from nbclient import NotebookClient
    from nbclient.exceptions import CellExecutionError

    nb = nbformat.read(path, as_version=4)
    kept, dropped = [], 0
    for cell in nb.cells:
        if cell.cell_type == "code" and is_setup_cell("".join(cell.source)):
            dropped += 1
            continue
        kept.append(cell)
    nb.cells = kept

    t0 = time.time()
    client = NotebookClient(
        nb,
        timeout=CELL_TIMEOUT,
        kernel_name="python3",
        resources={"metadata": {"path": str(REPO)}},
        allow_errors=False,
    )
    try:
        client.execute()
    except CellExecutionError as exc:
        return False, _trim(str(exc)), time.time() - t0
    except Exception as exc:                        # kernel died, timeout, ...
        return False, f"{type(exc).__name__}: {_trim(str(exc))}", time.time() - t0
    note = f"{len(kept)} cells" + (f", {dropped} setup cell(s) skipped" if dropped else "")
    return True, note, time.time() - t0


def run_tutorial(path: Path) -> tuple[bool, str, float]:
    env = dict(os.environ, MPLBACKEND="Agg", PYTHONWARNINGS="ignore")
    (REPO / "tutorial_figures").mkdir(exist_ok=True)   # tutorials savefig into it
    t0 = time.time()
    try:
        proc = subprocess.run([sys.executable, str(path)], cwd=REPO, env=env,
                              capture_output=True, text=True,
                              timeout=TUTORIAL_TIMEOUT)
    except subprocess.TimeoutExpired:
        return False, f"timed out after {TUTORIAL_TIMEOUT}s", time.time() - t0
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout).strip().splitlines()
        return False, _trim("\n".join(tail[-6:])), time.time() - t0
    return True, "", time.time() - t0


def _trim(text: str, limit: int = 600) -> str:
    text = text.strip()
    return text if len(text) <= limit else text[:limit] + " ...[trimmed]"


# --------------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--notebooks-only", action="store_true")
    ap.add_argument("--tutorials-only", action="store_true")
    ap.add_argument("--list", action="store_true", help="classify, run nothing")
    args = ap.parse_args()

    nbs = notebooks()
    runnable, wave_dependent = tutorials()

    if args.list:
        print(f"notebooks ({len(nbs)}):")
        for p in nbs:
            print(f"   {p.name}")
        print(f"\ntutorials, runnable ({len(runnable)}):")
        for p in runnable:
            print(f"   {p.name}")
        print(f"\ntutorials, need wave/ ({len(wave_dependent)}):")
        for p in wave_dependent:
            print(f"   {p.name}")
        return 0

    failures: list[tuple[str, str]] = []

    if not args.tutorials_only:
        print(f"=== notebooks ({len(nbs)}) " + "=" * 40)
        if not nbs:
            print("   none found -- nothing to verify, which is itself suspicious")
        for path in nbs:
            ok, note, secs = run_notebook(path)
            print(f"   {'PASS' if ok else 'FAIL'}  {path.name:<34} {secs:6.1f}s  {note if ok else ''}")
            if not ok:
                failures.append((path.name, note))

    if not args.notebooks_only:
        print(f"\n=== tutorials, runnable ({len(runnable)}) " + "=" * 26)
        for path in runnable:
            ok, note, secs = run_tutorial(path)
            print(f"   {'PASS' if ok else 'FAIL'}  {path.name:<44} {secs:6.1f}s")
            if not ok:
                failures.append((path.name, note))

    # Reported, never hidden. These are the ones an outsider cannot run at all.
    print(f"\n=== tutorials NOT verified -- require wave/ ({len(wave_dependent)}) ===")
    print("    wave/ is gitignored, so these do not run from a fresh clone.")
    for path in wave_dependent:
        print(f"   UNVERIFIED  {path.name}")

    print("\n" + "=" * 62)
    if failures:
        print(f"FAILED: {len(failures)}")
        for name, note in failures:
            print(f"\n--- {name} ---\n{note}")
        return 1
    verified = (0 if args.tutorials_only else len(nbs)) + (0 if args.notebooks_only else len(runnable))
    print(f"OK: {verified} verified, {len(wave_dependent)} unverified (need wave/)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
