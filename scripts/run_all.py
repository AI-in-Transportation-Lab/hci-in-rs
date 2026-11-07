#!/usr/bin/env python3
"""Run the book's case studies and collect their figures under outputs/.

Each script is executed with outputs/chNN/ as the working directory, so the
figures and files the listings write with relative paths land there instead
of in the source tree.

Examples
--------
    python scripts/run_all.py                 # every case study
    python scripts/run_all.py --chapter 6     # one chapter
    python scripts/run_all.py -c 3 -k 2       # chapter 3, case study 2
    python scripts/run_all.py --list          # show what would run
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CHAPTERS = ROOT / "chapters"
OUTPUTS = ROOT / "outputs"

# cs<N>_<topic>.py is case study N; we<N>_<topic>.py is a worked example.
SCRIPT_RE = re.compile(r"^(cs|we)(\d+)_.*\.py$")
CHAPTER_RE = re.compile(r"^ch(\d+)_")


def discover(chapter: int | None, case: int | None) -> list[tuple[int, str, Path]]:
    found = []
    for ch_dir in sorted(p for p in CHAPTERS.iterdir() if p.is_dir()):
        m = CHAPTER_RE.match(ch_dir.name)
        if not m:
            continue
        ch = int(m.group(1))
        if chapter is not None and ch != chapter:
            continue
        for script in sorted(ch_dir.glob("*.py")):
            sm = SCRIPT_RE.match(script.name)
            if sm and (case is None or (sm.group(1) == "cs" and int(sm.group(2)) == case)):
                found.append((ch, sm.group(1) + sm.group(2), script))
    return found


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-c", "--chapter", type=int, help="chapter number (1-12)")
    ap.add_argument("-k", "--case", type=int, help="case study number within the chapter (1 or 2)")
    ap.add_argument("--list", action="store_true", help="list matching scripts and exit")
    args = ap.parse_args()

    scripts = discover(args.chapter, args.case)
    if not scripts:
        print("No matching case studies.")
        return 1
    if args.list:
        for ch, tag, path in scripts:
            print(f"ch{ch:02d} {tag}  {path.relative_to(ROOT)}")
        return 0

    failed = []
    for ch, tag, path in scripts:
        out_dir = OUTPUTS / f"ch{ch:02d}"
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"== ch{ch:02d} {tag}: {path.name}", flush=True)
        t0 = time.time()
        proc = subprocess.run([sys.executable, str(path)], cwd=out_dir,
                              env={**os.environ, "MPLBACKEND": "Agg"})
        status = "ok" if proc.returncode == 0 else f"FAILED (exit {proc.returncode})"
        print(f"   {status} in {time.time() - t0:.1f}s\n", flush=True)
        if proc.returncode:
            failed.append(path.relative_to(ROOT))

    print(f"{len(scripts) - len(failed)}/{len(scripts)} scripts succeeded.")
    for p in failed:
        print(f"  failed: {p}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
