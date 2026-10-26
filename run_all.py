from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parent
    for script in [
        "analysis_primary.py",
        "analysis_trial_secondary.py",
        "analysis_subjective.py",
    ]:
        subprocess.run(
            [sys.executable, str(root / script), "--input", str(args.input), "--output", str(args.output)],
            check=True,
        )


if __name__ == "__main__":
    main()
