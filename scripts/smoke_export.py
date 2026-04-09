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
    # Determine script's own directory based on known repo structure:
    # scripts/smoke_export.py is at <repo>/scripts/smoke_export.py
    # HERMES_HOME = <repo>, so script_path = HERMES_HOME / "scripts" / "smoke_export.py"
    script_path = hermes_home / "scripts" / "smoke_export.py"

    config = hermes_home / "config.yaml"
    config.write_text("model: test\n", encoding="utf-8")

    bundle = hermes_home / "smoke-bundle.tar.gz"
    result = subprocess.run(
        [sys.executable, "-m", "hermes", "migrate", "export", "--output", str(bundle)],
        check=False,
        capture_output=True,
        cwd=str(hermes_home),
    )
    if result.returncode != 0:
        if result.stdout:
            print(result.stdout.decode("utf-8", errors="replace"))
        if result.stderr:
            print(result.stderr.decode("utf-8", errors="replace"))
        raise SystemExit(result.returncode)
    print(f"Bundle created at {bundle}")


if __name__ == "__main__":
    main()
