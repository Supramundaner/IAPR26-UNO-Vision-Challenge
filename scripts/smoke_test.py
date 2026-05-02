from __future__ import annotations

import subprocess
import sys


def run(command: list[str]) -> None:
    print("$ " + " ".join(command))
    subprocess.run(command, check=True)


def main() -> None:
    run([sys.executable, "scripts/check_models.py", "--image-size", "128"])
    run(
        [
            sys.executable,
            "scripts/generate_submission.py",
            "--image-size",
            "128",
            "--allow-random",
            "--output",
            "submissions/smoke_submission.csv",
        ]
    )


if __name__ == "__main__":
    main()

