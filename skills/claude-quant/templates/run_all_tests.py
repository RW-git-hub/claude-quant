"""
run_all_tests.py - discover and run every template's self-tests.

Every template in this directory is self-testing: its `if __name__ == "__main__"`
block runs assert-based checks against analytic or synthetic cases. This harness
runs them all as subprocesses and prints a pass/fail summary. Exit code is
non-zero if any template fails, so it doubles as a CI gate.

Usage:  python run_all_tests.py
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

SKIP = {"run_all_tests.py", "__init__.py"}


def main() -> int:
    here = Path(__file__).resolve().parent
    files = sorted(p for p in here.glob("*.py") if p.name not in SKIP)
    if not files:
        print("no templates found")
        return 1

    results = []
    for f in files:
        proc = subprocess.run(
            [sys.executable, str(f)], capture_output=True, text=True
        )
        ok = proc.returncode == 0
        stream = proc.stdout if ok else (proc.stderr or proc.stdout)
        last = (stream.strip().splitlines() or [""])[-1]
        results.append((f.name, ok, last))

    width = max(len(n) for n, _, _ in results)
    print("=" * (width + 80))
    n_pass = 0
    for name, ok, last in results:
        if ok:
            n_pass += 1
        tag = "PASS" if ok else "FAIL"
        print(f"{tag}  {name.ljust(width)}  {last[:72]}")
    print("=" * (width + 80))
    print(f"{n_pass}/{len(results)} template self-tests passed")
    return 0 if n_pass == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
