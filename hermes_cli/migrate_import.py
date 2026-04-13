"""
Hermes migration import command.

Imports a migration bundle into current Hermes installation.
"""

import json
import os
import shutil
import sys
import tarfile
from pathlib import Path
from typing import Optional

import yaml

from hermes_cli.colors import Colors, color
from hermes_cli.migrate_core import (
    HERMES_HOME,
    MigrationReport,
    _EXCLUDE_ALWAYS,
    _EXTERNAL_TOOLS,
    _PLATFORM_SKIP,
    _is_secret,
    _is_text_file,
    _log_error,
    _log_warning,
    _remap_content,
    detect_platform,
)
from hermes_cli.profiles import _safe_extract_profile_archive


def import_bundle(
    input_path: str,
    dry_run: bool = False,
    interactive: bool = False,
) -> MigrationReport:
    """Import a migration bundle into current Hermes installation.

    Args:
        input_path: Path to .tar.gz bundle
        dry_run: If True, show what would be done without applying
        interactive: If True, run guided interactive mode

    Returns:
        MigrationReport with categorized items
    """
    bundle_path = Path(input_path)
    if not bundle_path.exists():
        raise FileNotFoundError(f"Bundle not found: {bundle_path}")

    manifest = _read_manifest(bundle_path)
    preset = manifest.get("preset", "safe")
    source_platform = {
        "os": manifest.get("source_os", "unknown"),
        "home": Path(manifest.get("source_home", "~")),
    }
    target_platform = detect_platform()

    print()
    print(color("┌─────────────────────────────────────────────────────────┐", Colors.MAGENTA))
    print(color("│          Hermes Migration — Import                       │", Colors.MAGENTA))
    print(color("└─────────────────────────────────────────────────────────┘", Colors.MAGENTA))
    print()
    print(f"  Source:     {source_platform['os']} ({source_platform['home']})")
    print(f"  Target:     {target_platform['os']} ({target_platform['home']})")
    print(f"  Bundle:     {bundle_path.name}")
    print(f"  Preset:     {manifest.get('preset', preset)}")
    print()

    source_home = source_platform["home"]
    target_home = target_platform["home"]

    if str(source_home) != str(target_home):
        print(color("  Path remapping:", Colors.CYAN))
        print(f"    {source_home} → {target_home}")
        print()

    # Discover what is in the bundle
    report = _discover_bundle_contents(bundle_path, target_platform)

    if dry_run:
        print(color("  [DRY RUN] No changes made", Colors.YELLOW))
        _show_bundle_contents(bundle_path)
        _print_migration_report(report, manifest, target_platform)
        return report

    if interactive:
        _run_interactive(bundle_path, source_home, target_home, manifest, target_platform, report)
        return report

    # Auto-import path
    print(color("  This will merge migration data into your current Hermes installation.", Colors.YELLOW))
    print()

    _backup_conflicts(bundle_path)

    print(color("  Extracting bundle...", Colors.CYAN))
    _extract_with_remap(bundle_path, source_home, target_home, report)

    _remap_config_paths(source_home, target_home, manifest)

    _post_import_verify(report, manifest, target_platform)

    print()
    print(color("✓ Migration complete!", Colors.GREEN))
    print()
    _print_migration_report(report, manifest, target_platform)

    return report


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


def _discover_bundle_contents(bundle_path: Path, target_platform: dict) -> MigrationReport:
    """Discover what's in a bundle and categorize each item."""
    report = MigrationReport()
    target_os = target_platform["os"]

    with tarfile.open(bundle_path, "r:gz") as tf:
        for member in tf.getmembers():
            if member.name == "manifest.json":
                continue
            name = member.name

            # Skip always-excluded files
            if name in _EXCLUDE_ALWAYS:
                report.skipped.append(f"{name}  [runtime artifact, skipped]")
                continue

            # Check for platform incompatibility
            if target_os in _PLATFORM_SKIP:
                ext = os.path.splitext(name)[1].lower()
                if ext in _PLATFORM_SKIP.get(target_os, set()):
                    report.incompatible.append(
                        f"{name}  [{target_os} incompatible — skipped]"
                    )
                    continue

            # Categorize by type
            if member.isdir() or name.endswith("/"):
                report.migrated.append(f"{name}/")
            else:
                report.migrated.append(name)

    return report


def _show_bundle_contents(bundle_path: Path) -> None:
    """Show what files are in the bundle."""
    print(color("  Bundle contents:", Colors.CYAN))
    with tarfile.open(bundle_path, "r:gz") as tf:
        for name in sorted(tf.getnames()):
            if name == "manifest.json":
                continue
            info = tf.getmember(name)
            if info.isdir():
                print(f"    📁 {name}/")
            else:
                size = info.size // 1024
                print(f"    📄 {name} ({size} KB)")
    print()


def _backup_conflicts(bundle_path: Path) -> None:
    """Backup files that would be overwritten during import."""
    conflicts = []
    with tarfile.open(bundle_path, "r:gz") as tf:
        for member in tf.getmembers():
            if member.name == "manifest.json":
                continue
            dest = HERMES_HOME / member.name
            if dest.exists():
                conflicts.append(member.name)

    if not conflicts:
        return

    from datetime import datetime
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    backup_dir = HERMES_HOME / f".hermes.bak.{ts}"
    backup_dir.mkdir(exist_ok=True)

    for name in conflicts:
        src = HERMES_HOME / name
        dst = backup_dir / name
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.is_dir():
            shutil.copytree(src, dst, dirs_exist_ok=True)
        else:
            shutil.copy2(src, dst)

    print(color(f"  ⚠ {len(conflicts)} files backed up to {backup_dir.name}/", Colors.YELLOW))


def _extract_with_remap(
    bundle_path: Path,
    source_home: Path,
    target_home: Path,
    report: MigrationReport,
) -> None:
    """Extract bundle with home path remapping and report population."""
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)

        _safe_extract_profile_archive(bundle_path, tmppath)

        extracted_items = []
        for item in tmppath.iterdir():
            rel = item.name
            if rel == "manifest.json":
                continue
            if _is_secret(rel, preset):
                report.skipped.append(f"{rel}  [secrets excluded in safe preset]")
                continue

            dest = HERMES_HOME / rel
            extracted_items.append(rel)

            if item.is_file() and _is_text_file(rel):
                content = item.read_text(encoding="utf-8", errors="replace")
                remapped_content = _remap_content(content, source_home, target_home)
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_text(remapped_content, encoding="utf-8")
            elif item.is_dir():
                # Recursively copy directory, remapping text file contents
                for src_sub in item.rglob("*"):
                    rel_sub = src_sub.relative_to(item)
                    dest_sub = dest / rel_sub
                    # Security: prevent path traversal attacks
                    dest_sub_resolved = dest_sub.resolve()
                    if not str(dest_sub_resolved).startswith(str(HERMES_HOME.resolve())):
                        _log_warning(f"Skipping path traversal attempt: {dest_sub}")
                        continue
                    if src_sub.is_symlink():
                        _log_warning(f"Skipping symlink: {src_sub}")
                        continue
                    if src_sub.is_file():
                        dest_sub.parent.mkdir(parents=True, exist_ok=True)
                        if _is_secret(str(src_sub.relative_to(item)), preset):
                            continue
                        if _is_text_file(src_sub.name):
                            file_content = src_sub.read_text(encoding="utf-8", errors="replace")
                            remapped = _remap_content(file_content, source_home, target_home)
                            dest_sub.write_text(remapped, encoding="utf-8")
                        else:
                            shutil.copy2(src_sub, dest_sub)
                    else:
                        dest_sub.mkdir(parents=True, exist_ok=True)
                # Ensure dest dir itself exists (for empty dirs)
                dest.mkdir(parents=True, exist_ok=True)
            else:
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(item, dest)


def _remap_config_paths(source_home: Path, target_home: Path, manifest: dict) -> None:
    """Remap absolute paths inside config.yaml."""
    config_path = HERMES_HOME / "config.yaml"
    if not config_path.exists():
        return

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        if not config:
            return

        modified = False
        source_str = str(source_home)
        target_str = str(target_home)

        terminal = config.get("terminal", {})
        cwd = terminal.get("cwd", "")

        if cwd and cwd.startswith(source_str):
            terminal["cwd"] = cwd.replace(source_str, target_str, 1)
            config["terminal"] = terminal
            modified = True

        docker_volumes = terminal.get("docker_volumes", [])
        if docker_volumes:
            new_volumes = []
            for vol in docker_volumes:
                if ":" in vol:
                    host_path, sep, container = vol.partition(":")
                    if host_path.startswith(source_str):
                        new_host = target_str + host_path[len(source_str):]
                        new_vol = new_host + sep + container
                        new_volumes.append(new_vol)
                        modified = True
                    else:
                        new_volumes.append(vol)
                else:
                    new_volumes.append(vol)
            if modified:
                terminal["docker_volumes"] = new_volumes
                config["terminal"] = terminal

        external_dirs = config.get("external_dirs", [])
        if external_dirs:
            new_dirs = []
            for d in external_dirs:
                if d.startswith(source_str):
                    new_dirs.append(d.replace(source_str, target_str, 1))
                    modified = True
                else:
                    new_dirs.append(d)
            if modified:
                config["external_dirs"] = new_dirs

        if modified:
            with open(config_path, "w", encoding="utf-8") as f:
                yaml.safe_dump(config, f, default_flow_style=False, allow_unicode=True)
            print(color("  ✓ Config paths remapped", Colors.GREEN))
        else:
            print(color("  ✓ No config paths to remap", Colors.GREEN))

    except Exception as e:
        print(color(f"  ⚠ Could not remap config paths: {e}", Colors.YELLOW))


def _post_import_verify(
    report: MigrationReport,
    manifest: dict,
    target_platform: dict,
) -> None:
    """Run post-import verification."""
    target_os = target_platform["os"]
    target_home = target_platform["home"]

    print()
    print(color("  Running post-import verification...", Colors.CYAN))

    # Provider / auth check
    auth_issues = _verify_providers(target_home)
    for item in auth_issues:
        report.needs_reauth.append(item)

    # Path existence check for remapped paths
    path_issues = _verify_remapped_paths(target_home)
    for item in path_issues:
        report.needs_reauth.append(item)

    # External tool availability
    tool_issues = _verify_external_tools()
    for item in tool_issues:
        report.incompatible.append(item)

    # Platform-specific binding warnings
    platform_issues = _verify_platform_bindings(target_os, target_home)
    for item in platform_issues:
        report.incompatible.append(item)

    # Print results
    if report.needs_reauth:
        print(color("  ⚠ Needs manual re-auth:", Colors.YELLOW))
        for item in report.needs_reauth:
            print(f"    • {item}")
        print()

    if report.incompatible:
        print(color("  ✗ Incompatible with target environment:", Colors.RED))
        for item in report.incompatible:
            print(f"    • {item}")
        print()


def _verify_providers(target_home: Path) -> list[str]:
    """Check if configured providers have corresponding auth entries."""
    issues = []

    config_path = HERMES_HOME / "config.yaml"
    auth_path = HERMES_HOME / "auth.json"

    if not config_path.exists():
        return issues

    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    providers = config.get("providers", {})

    auth_data = {}
    auth_corrupted = False
    if auth_path.exists():
        try:
            auth_data = json.loads(auth_path.read_text(encoding="utf-8"))
        except Exception as e:
            _log_error(f"auth.json is corrupted and will be skipped: {e}")
            auth_corrupted = True

    for provider_name, provider_config in providers.items():
        api_key = provider_config.get("api_key", "")
        if not api_key or api_key == "your-api-key-here":
            if auth_corrupted:
                continue
            auth_entry = auth_data.get("providers", {}).get(provider_name, {})
            if not auth_entry:
                issues.append(
                    f"provider '{provider_name}' — no API key configured, "
                    f"run 'hermes config set providers.{provider_name}.api_key <key>'"
                )

    return issues


def _verify_remapped_paths(target_home: Path) -> list[str]:
    """Check that paths referenced in config actually exist on target."""
    issues = []

    config_path = HERMES_HOME / "config.yaml"
    if not config_path.exists():
        return issues

    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}

    terminal = config.get("terminal", {})
    cwd = terminal.get("cwd", "")
    if cwd and not Path(cwd).exists():
        issues.append(f"working directory does not exist: {cwd}")

    for d in config.get("external_dirs", []):
        if d and not Path(d).exists():
            issues.append(f"external directory not found: {d}  (create or update config)")

    for vol in terminal.get("docker_volumes", []):
        if ":" in vol:
            host_path = vol.split(":")[0]
            if host_path and not Path(host_path).exists():
                issues.append(f"docker volume host path not found: {host_path}")

    return issues


def _verify_external_tools() -> list[str]:
    """Check if external tools referenced in config are available."""
    issues = []

    config_path = HERMES_HOME / "config.yaml"
    if not config_path.exists():
        return issues

    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}

    terminal = config.get("terminal", {})
    aliases = terminal.get("aliases", {})

    for name, cmd in aliases.items():
        if isinstance(cmd, str):
            tool = cmd.split()[0]
            if tool in _EXTERNAL_TOOLS:
                if not shutil.which(tool):
                    issues.append(
                        f"alias '{name}' uses '{tool}' which is not installed on this system"
                    )

    return issues


def _verify_platform_bindings(target_os: str, target_home: Path) -> list[str]:
    """Detect platform-specific bindings that may not work on target."""
    issues = []

    config_path = HERMES_HOME / "config.yaml"
    if not config_path.exists():
        return issues

    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}

    # WSL-specific: warn about /mnt/c paths on non-WSL
    if target_os != "wsl":
        terminal = config.get("terminal", {})
        cwd = terminal.get("cwd", "")
        if cwd and "/mnt/" in cwd:
            issues.append(
                f"working directory uses WSL /mnt/ path: {cwd}  "
                f"(WSL-specific, may not work on {target_os})"
            )

    # Check for macOS-only features on Linux/WSL
    if target_os in ("linux", "wsl"):
        toolchain = config.get("toolchain", {})
        shell = toolchain.get("shell", "")
        if shell in ("/bin/zsh", "/opt/homebrew/bin/zsh"):
            issues.append(
                f"shell '{shell}' is macOS-specific  "
                f"(fallback shell will be used)"
            )

    return issues


def _print_migration_report(
    report: MigrationReport,
    manifest: dict,
    target_platform: dict,
) -> None:
    """Print a categorized migration report."""
    source_os = manifest.get("source_os", "unknown")
    target_os = target_platform["os"]
    cross_platform = source_os != target_os and source_os != "unknown"

    print(color("  Migration Report", Colors.CYAN))
    print(color("  ─────────────────", Colors.CYAN))

    total = (
        len(report.migrated)
        + len(report.skipped)
        + len(report.needs_reauth)
        + len(report.incompatible)
    )
    print(f"  Total items:  {total}")
    print(f"  {color('✓', Colors.GREEN)} Migrated:       {len(report.migrated)}")
    if report.skipped:
        print(f"  {color('⚠', Colors.YELLOW)} Skipped:        {len(report.skipped)}")
    if report.needs_reauth:
        print(f"  {color('⚠', Colors.YELLOW)} Needs re-auth:  {len(report.needs_reauth)}")
    if report.incompatible:
        print(f"  {color('✗', Colors.RED)} Incompatible:  {len(report.incompatible)}")
    print()

    if cross_platform:
        print(color(f"  ⚠ Cross-platform migration: {source_os} → {target_os}", Colors.YELLOW))
        print()

    if report.migrated:
        print(color("  ✓ Migrated:", Colors.GREEN))
        for item in report.migrated[:20]:
            print(f"      {item}")
        if len(report.migrated) > 20:
            print(f"      ... and {len(report.migrated) - 20} more")
        print()

    if report.skipped:
        print(color("  ⚠ Skipped:", Colors.YELLOW))
        for item in report.skipped[:10]:
            print(f"      {item}")
        if len(report.skipped) > 10:
            print(f"      ... and {len(report.skipped) - 10} more")
        print()

    if report.needs_reauth:
        print(color("  ⚠ Needs manual re-authentication:", Colors.YELLOW))
        for item in report.needs_reauth:
            print(f"      • {item}")
        print()
        print("    Run 'hermes config migrate' to update credentials.")
        print()

    if report.incompatible:
        print(color("  ✗ Incompatible with target environment:", Colors.RED))
        for item in report.incompatible:
            print(f"      • {item}")
        print()
        print("    These items were skipped. Check the items above for alternatives.")
        print()

    if not report.needs_reauth and not report.incompatible:
        if manifest.get("includes_secrets"):
            print(color("  🔐 Secrets bundle — credentials included", Colors.GREEN))
        else:
            print(color("  ✓ No re-authentication needed", Colors.GREEN))
        print()

    print("  Next steps:")
    print(color("    hermes migrate verify", Colors.CYAN))
    print(color("    hermes migrate doctor", Colors.CYAN))
    print()


def _run_interactive(
    bundle_path: Path,
    source_home: Path,
    target_home: Path,
    manifest: dict,
    target_platform: dict,
    report: MigrationReport,
) -> None:
    """Guided interactive migration flow."""
    print(color("\n  Interactive mode — follow the prompts\n", Colors.CYAN))

    # Step 1: Show bundle summary
    print(color("  Step 1: Bundle contents", Colors.CYAN))
    print(color("  ─────────────────────────", Colors.CYAN))
    _show_bundle_contents(bundle_path)

    # Step 2: Summary
    total_items = sum(len(lst) for lst in [
        report.migrated, report.skipped, report.incompatible
    ])
    print(f"  Found {total_items} items: "
          f"{len(report.migrated)} to migrate, "
          f"{len(report.skipped)} skipped, "
          f"{len(report.incompatible)} incompatible")
    print()

    # Step 3: Ask about secrets
    if not manifest.get("includes_secrets"):
        print(color("  Step 2: Secrets", Colors.CYAN))
        print(color("  ─────────────────", Colors.CYAN))
        print("  This bundle does NOT include secrets (.env, auth.json).")
        print("  After import, you'll need to re-enter API keys manually.")
        print("  To include secrets next time, run: hermes migrate export --preset full")
        print()

    # Step 4: Path remapping
    if str(source_home) != str(target_home):
        print(color("  Step 3: Path remapping", Colors.CYAN))
        print(color("  ────────────────────────", Colors.CYAN))
        print("  Home directory paths will be remapped:")
        print(f"    {source_home} → {target_home}")
        print()

    # Step 5: Confirm
    print(color("  Step 4: Confirm", Colors.CYAN))
    print(color("  ─────────────────", Colors.CYAN))

    try:
        if sys.stdin.isatty():
            response = input("  Proceed with import? [y/N]  ").strip().lower()
        else:
            print(color("  Error: interactive mode requires a TTY (terminal).", Colors.RED))
            sys.exit(1)
    except KeyboardInterrupt:
        print(color("\n\nCancelled.", Colors.YELLOW))
        sys.exit(130)

    if response not in ("y", "yes"):
        print(color("\n  Import cancelled.", Colors.YELLOW))
        sys.exit(0)

    print()

    _backup_conflicts(bundle_path)

    print(color("  Extracting bundle...", Colors.CYAN))
    _extract_with_remap(bundle_path, source_home, target_home, report)

    _remap_config_paths(source_home, target_home, manifest)

    _post_import_verify(report, manifest, target_platform)

    print()
    print(color("✓ Migration complete!", Colors.GREEN))
    print()
    _print_migration_report(report, manifest, target_platform)
