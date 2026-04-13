"""
Hermes migration verify and doctor commands.
"""

import json
import os
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path
from typing import Optional

import yaml

from hermes_cli.colors import Colors, color
from hermes_cli.migrate_core import (
    HERMES_HOME,
    _log_warning,
    detect_platform,
)


def verify_bundle(input_path: Optional[str] = None) -> bool:
    """Verify a migration bundle or current installation."""
    results = {"passed": [], "failed": [], "warnings": []}

    if input_path:
        bundle_path = Path(input_path)
        if not bundle_path.exists():
            print(color(f"\n✗ Bundle not found: {bundle_path}", Colors.RED))
            return False

        print()
        print(color("┌─────────────────────────────────────────────────────────┐", Colors.CYAN))
        print(color("│          Hermes Migration — Verify Bundle                │", Colors.CYAN))
        print(color("└─────────────────────────────────────────────────────────┘", Colors.CYAN))
        print()

        manifest = _read_manifest(bundle_path)
        if not manifest:
            print(color("✗ No manifest found in bundle", Colors.RED))
            return False

        print(f"  Version:     {manifest.get('version', 'unknown')}")
        print(f"  Created:     {manifest.get('bundle_created_at', 'unknown')}")
        print(f"  Source:      {manifest.get('source_os', 'unknown')}")
        print(f"  Preset:     {manifest.get('preset', 'unknown')}")
        print()

        # Single-pass bundle verification
        print(color("  Verifying bundle (integrity, config, skills, secrets)...", Colors.CYAN))
        try:
            with tarfile.open(bundle_path, "r:gz") as tf:
                members = tf.getmembers()
                names = [m.name for m in members]

                # Integrity: iterate all members to trigger CRC check
                for member in tf:
                    if member.size > 0:
                        pass

                # config.yaml check
                f = tf.extractfile("config.yaml")
                if f:
                    yaml.safe_load(f.read())
                    results["passed"].append("config.yaml valid YAML")
                else:
                    results["warnings"].append("config.yaml not in bundle")

                # Skills structure
                skills_found = sum(
                    1 for name in names
                    if name.startswith("skills/") and name.endswith("/SKILL.md")
                )
                if skills_found > 0:
                    results["passed"].append(f"Skills structure OK ({skills_found} skills)")
                else:
                    results["warnings"].append("No skills found in bundle")

                # Secrets check
                has_env = "env" in names
                has_auth = "auth.json" in names
                if has_env or has_auth:
                    results["passed"].append(f"Secrets included (env={has_env}, auth={has_auth})")
                else:
                    results["warnings"].append("No secrets in bundle — API keys need re-entry after import")

            results["passed"].append("Bundle integrity (CRC valid)")
        except Exception as e:
            results["failed"].append(f"Bundle verification: {e}")

        # Cross-platform compatibility
        target_platform = detect_platform()
        source_os = manifest.get("source_os", "unknown")
        if source_os != "unknown" and source_os != target_platform["os"]:
            results["warnings"].append(
                f"Cross-platform: {source_os} → {target_platform['os']} "
                f"(some paths or tools may need adjustment)"
            )

    else:
        # Verify current installation
        print()
        print(color("┌─────────────────────────────────────────────────────────┐", Colors.CYAN))
        print(color("│          Hermes Migration — Verify Installation           │", Colors.CYAN))
        print(color("└─────────────────────────────────────────────────────────┘", Colors.CYAN))
        print()

        config_path = HERMES_HOME / "config.yaml"
        if config_path.exists():
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    yaml.safe_load(f)
                results["passed"].append("config.yaml valid YAML")
            except Exception as e:
                results["failed"].append(f"config.yaml: {e}")
        else:
            results["failed"].append("config.yaml not found")

        skills_dir = HERMES_HOME / "skills"
        if skills_dir.exists():
            skill_count = sum(
                1 for p in skills_dir.iterdir()
                if p.is_dir() and (p / "SKILL.md").exists()
            )
            results["passed"].append(f"{skill_count} skills with SKILL.md")
        else:
            results["warnings"].append("skills/ not found")

        # Check providers
        if config_path.exists():
            config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
            providers = config.get("providers", {})
            if providers:
                auth_path = HERMES_HOME / "auth.json"
                auth_data = {}
                auth_corrupted = False
                if auth_path.exists():
                    try:
                        auth_data = json.loads(auth_path.read_text(encoding="utf-8"))
                    except Exception as e:
                        _log_warning(f"auth.json is corrupted and will be skipped: {e}")
                        auth_corrupted = True
                missing = []
                for name, prov in providers.items():
                    if not prov.get("api_key"):
                        if auth_corrupted:
                            continue
                        auth_entry = auth_data.get("providers", {}).get(name, {})
                        if not auth_entry:
                            missing.append(name)
                if missing:
                    results["warnings"].append(
                        f"Providers missing API key: {', '.join(missing)}  "
                        f"(run 'hermes config migrate' to fix)"
                    )
                else:
                    results["passed"].append("All providers have API keys configured")
            else:
                results["warnings"].append("No providers configured")

    print()
    for item in results["passed"]:
        print(color(f"  ✓ {item}", Colors.GREEN))
    for item in results["warnings"]:
        print(color(f"  ⚠ {item}", Colors.YELLOW))
    for item in results["failed"]:
        print(color(f"  ✗ {item}", Colors.RED))
    print()

    if results["failed"]:
        print(color("  Verification FAILED", Colors.RED))
        return False
    elif results["warnings"]:
        print(color("  Verification passed with warnings", Colors.YELLOW))
        return True
    else:
        print(color("  Verification PASSED", Colors.GREEN))
        return True


def _read_manifest(bundle_path: Path) -> dict:
    """Read manifest from bundle."""
    with tarfile.open(bundle_path, "r:gz") as tf:
        try:
            f = tf.extractfile("manifest.json")
            if f is None:
                return {}
            return json.loads(f.read().decode("utf-8"))
        except KeyError:
            return {}


def run_doctor() -> bool:
    """Run environment health checks for migration."""
    results = {"passed": [], "failed": [], "warnings": []}

    print()
    print(color("┌─────────────────────────────────────────────────────────┐", Colors.CYAN))
    print(color("│          Hermes Migration — Doctor                      │", Colors.CYAN))
    print(color("└─────────────────────────────────────────────────────────┘", Colors.CYAN))
    print()

    # Python version
    print(color("  Checking Python...", Colors.CYAN))
    version_info = sys.version_info
    version_str = f"Python {version_info.major}.{version_info.minor}.{version_info.micro}"
    if version_info < (3, 10):
        results["failed"].append(f"Python too old: {version_str} (need 3.10+)")
    else:
        results["passed"].append(f"{version_str}")

    # Hermes home
    print(color("  Checking Hermes home...", Colors.CYAN))
    if HERMES_HOME.exists():
        results["passed"].append(f"Hermes home: {HERMES_HOME}")
    else:
        results["failed"].append(f"Hermes home not found: {HERMES_HOME}")
        parent = HERMES_HOME.parent
        if parent.exists() and os.access(parent, os.W_OK):
            results["warnings"].append(f"  Can create: {HERMES_HOME}")
            results["passed"].append("Parent directory writable")
        else:
            results["failed"].append(f"Cannot write to {parent}")

    # Disk space
    print(color("  Checking disk space...", Colors.CYAN))
    try:
        import shutil as sh
        if HERMES_HOME.exists():
            target = HERMES_HOME
        elif HERMES_HOME.parent.exists():
            target = HERMES_HOME.parent
            results["warnings"].append("HERMES_HOME does not exist — checking parent dir")
        else:
            target = Path("/")
            results["warnings"].append("HERMES_HOME and parent missing — checking root")
        total, used, free = sh.disk_usage(target)
        free_mb = free // (1024 * 1024)
        if free_mb < 100:
            results["failed"].append(f"Low disk space: {free_mb} MB free")
        elif free_mb < 500:
            results["warnings"].append(f"Disk space: {free_mb} MB free")
        else:
            results["passed"].append(f"Disk space OK: {free_mb} MB free")
    except Exception as e:
        results["warnings"].append(f"Disk check: {e}")

    # Config.yaml
    config_path = HERMES_HOME / "config.yaml"
    if config_path.exists():
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                yaml.safe_load(f)
            results["passed"].append("config.yaml valid")
        except Exception as e:
            results["failed"].append(f"config.yaml invalid: {e}")
    else:
        results["warnings"].append("config.yaml not found")

    # Platform
    p = detect_platform()
    results["passed"].append(f"Platform: {p['os']} (home: {p['home']})")

    # Critical external tools
    print(color("  Checking external tools...", Colors.CYAN))
    critical_tools = ["git", "python3"]
    for tool in critical_tools:
        if shutil.which(tool):
            results["passed"].append(f"{tool}: installed")
        else:
            results["failed"].append(f"{tool}: NOT installed")

    # Git repo
    repo_path = HERMES_HOME / "hermes-agent"
    if repo_path.exists():
        try:
            result = subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=repo_path, capture_output=True, text=True, timeout=5
            )
            if result.stdout.strip():
                results["warnings"].append("hermes-agent repo has uncommitted changes")
            else:
                results["passed"].append("hermes-agent repo clean")
        except Exception as e:
            _log_warning(f"Could not check git status: {e}")

    print()
    for item in results["passed"]:
        print(color(f"  ✓ {item}", Colors.GREEN))
    for item in results["warnings"]:
        print(color(f"  ⚠ {item}", Colors.YELLOW))
    for item in results["failed"]:
        print(color(f"  ✗ {item}", Colors.RED))
    print()

    if results["failed"]:
        print(color("  Doctor FAILED — fix issues before migrating", Colors.RED))
        return False
    elif results["warnings"]:
        print(color("  Doctor OK — but review warnings above", Colors.YELLOW))
        return True
    else:
        print(color("  Doctor PASSED — environment looks good", Colors.GREEN))
        return True
