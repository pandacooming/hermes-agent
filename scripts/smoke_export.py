#!/usr/bin/env python3
"""
Smoke-test script for hermes migrate export.
Used by .github/workflows/test-migrate.yml

Creates a minimal config.yaml in HERMES_HOME, then runs hermes migrate export.
This ensures the smoke bundle includes config.yaml regardless of platform.
"""
import pathlib
import subprocess
import sys
import os


def main():
    hermes_home = pathlib.Path(os.environ["HERMES_HOME"])
    config = hermes_home / "config.yaml"
    config.write_text("model: test\n", encoding="utf-8")

    bundle = hermes_home / "smoke-bundle.tar.gz"
    result = subprocess.run(
        [sys.executable, "-m", "hermes", "migrate", "export", "--output", str(bundle)],
        check=False,
    )
    if result.returncode != 0:
        print(result.stdout.decode("utf-8", errors="replace"))
        print(result.stderr.decode("utf-8", errors="replace"))
        raise SystemExit(result.returncode)
    print(f"Bundle created at {bundle}")


if __name__ == "__main__":
    main()
