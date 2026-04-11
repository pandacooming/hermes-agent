"""
Hermes unified migration command.

Supports cross-machine, cross-platform migration between
Linux, macOS, and WSL2 with automatic path remapping.

Usage::

    hermes migrate export                    Export to hermes-migration-{timestamp}.tar.gz
    hermes migrate export --preset full     Include secrets
    hermes migrate export -o backup.tar.gz  Custom output path
    hermes migrate import -i backup.tar.gz  Import from bundle
    hermes migrate import -i backup.tar.gz --dry-run   Preview without applying
    hermes migrate import -i backup.tar.gz --interactive  Guided import
    hermes migrate verify -i backup.tar.gz  Verify bundle
    hermes migrate verify                   Verify current installation
    hermes migrate doctor                   Check environment health
"""

import json
import os
import platform
import shutil
import subprocess
import sys
import tarfile
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from io import BytesIO
from pathlib import Path
from typing import Optional

import yaml

from hermes_cli.colors import Colors, color
from hermes_cli.profiles import _safe_extract_profile_archive

HERMES_HOME = Path(os.environ.get("HERMES_HOME", os.path.expanduser("~/.hermes")))
BUNDLE_VERSION = "1.0"

# Files/directories that are NEVER migrated (platform-specific or runtime)
_EXCLUDE_ALWAYS = frozenset({
    "hermes-agent",
    ".worktrees",
    "node_modules",
    "state.db",
    "state.db-shm",
    "state.db-wal",
    "hermes_state.db",
    "response_store.db",
    "response_store.db-shm",
    "response_store.db-wal",
    "gateway.pid",
    "gateway_state.json",
    "processes.json",
    "auth.lock",
    ".update_check",
    "errors.log",
    ".hermes_history",
    "__pycache__",
})

_SECRET_FILES = frozenset({".env", "auth.json"})

_PLATFORM_SKIP = {
    "windows": {".ps1", ".bat", ".cmd"},
    "linux": {".ps1", ".bat", ".cmd"},
    "macos": {".ps1", ".bat", ".cmd"},
    "wsl": {".ps1", ".bat", ".cmd"},
}

# External tool dependencies that may need manual setup on target
_EXTERNAL_TOOLS = frozenset({
    "docker", "git", "node", "npm", "python3", "pip3",
    "ffmpeg", "ssh", "scp", "rsync", "curl", "wget",
})


# ============================================================================
# Migration Report
# ============================================================================


class ItemStatus(str, Enum):
    MIGRATED = "migrated"
    SKIPPED = "skipped"
    NEEDS_REAUTH = "needs_reauth"
    INCOMPATIBLE = "incompatible"


@dataclass
class MigrationReport:
    """Structured report of a migration operation."""
    migrated: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    needs_reauth: list[str] = field(default_factory=list)
    incompatible: list[str] = field(default_factory=list)

    def is_empty(self) -> bool:
        return not any([
            self.migrated, self.skipped,
            self.needs_reauth, self.incompatible
        ])


# ============================================================================
# Platform detection
# ============================================================================


def detect_platform() -> dict:
    """Detect current platform and home directory."""
    system = platform.system().lower()
    if system == "windows":
        return {
            "os": "windows",
            "home": Path(os.environ.get("USERPROFILE", os.path.expanduser("~"))),
        }
    elif system == "darwin":
        return {
            "os": "macos",
            "home": Path("/Users") / os.environ.get("USER", os.environ.get("USERNAME", "unknown")),
        }
    else:
        kernel_release = platform.release().lower()
        if "microsoft" in kernel_release or "wsl" in kernel_release:
            home = Path(os.environ.get("HOME", os.path.expanduser("~")))
            return {"os": "wsl", "home": home}
        return {
            "os": "linux",
            "home": Path("/home") / os.environ.get("USER", os.environ.get("USERNAME", "unknown")),
        }


def get_home() -> Path:
    """Get current user's home directory."""
    return Path(os.path.expanduser("~"))


# ============================================================================
# Manifest
# ============================================================================


def create_manifest(preset: str, source_platform: dict) -> dict:
    """Create migration manifest metadata."""
    return {
        "version": BUNDLE_VERSION,
        "bundle_created_at": datetime.now(timezone.utc).isoformat(),
        "hermes_version": _get_hermes_version(),
        "source_os": source_platform["os"],
        "source_home": str(source_platform["home"]),
        "preset": preset,
        "includes_secrets": preset == "full",
    }


def _get_hermes_version() -> str:
    """Get installed Hermes version."""
    try:
        from hermes_cli import __version__
        return __version__
    except Exception:
        return "unknown"


# ============================================================================
# Bundle creation (export)
# ============================================================================


def export_bundle(output_path: Optional[str], preset: str = "safe") -> Path:
    """Create a migration bundle from current Hermes installation."""
    if not HERMES_HOME.exists():
        raise FileNotFoundError(f"Hermes home not found: {HERMES_HOME}")

    if output_path:
        output = Path(output_path)
    else:
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        output = Path(f"hermes-migration-{ts}.tar.gz")

    source_platform = detect_platform()
    manifest = create_manifest(preset, source_platform)

    items = _collect_migration_items(preset)
    migrated_count = 0

    with tarfile.open(output, "w:gz") as tf:
        # Add manifest
        manifest_bytes = json.dumps(manifest, indent=2).encode("utf-8")
        manifest_info = tarfile.TarInfo(name="manifest.json")
        manifest_info.size = len(manifest_bytes)
        tf.addfile(manifest_info, BytesIO(manifest_bytes))

        for rel_path, item_info in items.items():
            src = HERMES_HOME / rel_path
            if not src.exists():
                continue

            # Respect _collect_migration_items skip decisions (preset-based secrets)
            if item_info.get("status") == "skipped":
                continue

            try:
                if src.is_dir():
                    for parent, dirs, files in os.walk(src):
                        dirs[:] = [d for d in dirs if not _should_skip_dir(d, source_platform["os"])]
                        for fname in files:
                            if _should_skip_file(fname, source_platform["os"]):
                                continue
                            full_path = Path(parent) / fname
                            arcname = str(full_path.relative_to(HERMES_HOME))
                            tf.add(str(full_path), arcname=arcname)
                            migrated_count += 1
                else:
                    if _should_skip_file(rel_path, source_platform["os"]):
                        continue
                    tf.add(str(src), arcname=rel_path)
                    migrated_count += 1
            except (OSError, IOError):
                pass

    size_kb = output.stat().st_size // 1024

    print(color("\n✓ Bundle created", Colors.GREEN))
    print(f"  Path:    {output}")
    print(f"  Size:    {size_kb} KB")
    print(f"  Preset:  {preset}")
    print(f"  Source:  {source_platform['os']} ({source_platform['home']})")
    print(f"  Items:   {migrated_count} files/directories")
    print()
    print("Transfer this file to your new machine, then run:")
    print(color(f"  hermes migrate import -i {output.name}", Colors.CYAN))
    print()

    return output


def _collect_migration_items(preset: str) -> dict:
    """Collect list of items to migrate with their status."""
    items = {}

    for dirname in ["memories", "sessions", "skills", "profiles", "hooks", "cron"]:
        dest = HERMES_HOME / dirname
        items[dirname + "/"] = {
            "type": "directory",
            "status": "migrated" if dest.exists() else "skipped",
            "reason": "not found" if not dest.exists() else None,
        }

    for fname in ["config.yaml", "SOUL.md"]:
        dest = HERMES_HOME / fname
        items[fname] = {
            "type": "file",
            "status": "migrated" if dest.exists() else "skipped",
            "reason": "not found" if not dest.exists() else None,
        }

    for fname in [".env", "auth.json"]:
        dest = HERMES_HOME / fname
        if fname in _SECRET_FILES and preset != "full":
            items[fname] = {"type": "file", "status": "skipped", "reason": "secrets excluded (use --preset full)"}
        else:
            items[fname] = {
                "type": "file",
                "status": "migrated" if dest.exists() else "skipped",
                "reason": "not found" if not dest.exists() else None,
            }

    return items


def _should_skip_dir(name: str, os_type: str) -> bool:
    if name.startswith("."):
        return True
    if name in _EXCLUDE_ALWAYS:
        return True
    return False


def _should_skip_file(name: str, os_type: str) -> bool:
    # .env and auth.json are NOT skipped here — they are handled by
    # _collect_migration_items which respects --preset flag.
    if name.startswith(".") and name not in _SECRET_FILES:
        return True
    if name in _EXCLUDE_ALWAYS:
        return True
    ext = os.path.splitext(name)[1].lower()
    if ext in _PLATFORM_SKIP.get(os_type, set()):
        return True
    return False


# ============================================================================
# Bundle import
# ============================================================================


def import_bundle(
    input_path: str,
    preset: str = "safe",
    dry_run: bool = False,
    interactive: bool = False,
) -> MigrationReport:
    """Import a migration bundle into current Hermes installation.

    Args:
        input_path: Path to .tar.gz bundle
        preset: "safe" or "full"
        dry_run: If True, show what would be done without applying
        interactive: If True, run guided interactive mode

    Returns:
        MigrationReport with categorized items
    """
    bundle_path = Path(input_path)
    if not bundle_path.exists():
        raise FileNotFoundError(f"Bundle not found: {bundle_path}")

    manifest = _read_manifest(bundle_path)
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

            dest = HERMES_HOME / rel
            extracted_items.append(rel)

            if item.is_file() and _is_text_file(rel):
                content = item.read_text(encoding="utf-8", errors="replace")
                remapped_content = _remap_content(content, source_home, target_home)
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_text(remapped_content, encoding="utf-8")
            elif item.is_dir():
                # Recursively copy directory, remapping text file contents
                # (shutil.copytree skips perms issues, rglob handles nesting)
                for src_sub in item.rglob("*"):
                    rel_sub = src_sub.relative_to(item)
                    dest_sub = dest / rel_sub
                    if src_sub.is_file():
                        dest_sub.parent.mkdir(parents=True, exist_ok=True)
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

        # Update report: anything not extracted was skipped
        for item in extracted_items:
            if item not in report.migrated and item not in report.skipped:
                report.migrated.append(item)


def _is_text_file(name: str) -> bool:
    """Check if file is a text file that may contain paths."""
    text_extensions = {".yaml", ".yml", ".json", ".md", ".txt", ".sh", ".env", ".toml", ".ini", ".cfg"}
    return os.path.splitext(name)[1].lower() in text_extensions or name in {".env", "config.yaml", "SOUL.md"}


def _remap_content(content: str, source_home: Path, target_home: Path) -> str:
    """Remap home directory paths in text content."""
    # Normalize to forward slashes for cross-platform compatibility
    source_str = str(source_home).replace("\\", "/")
    target_str = str(target_home).replace("\\", "/")
    if source_str != target_str and source_str in content:
        return content.replace(source_str, target_str)
    return content


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
                    host_path = vol.split(":")[0]
                    if host_path.startswith(source_str):
                        new_vol = vol.replace(source_str, target_str, 1)
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


# ============================================================================
# Bundle manifest reader
# ============================================================================


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


# ============================================================================
# Post-import verification
# ============================================================================


def _post_import_verify(
    report: MigrationReport,
    manifest: dict,
    target_platform: dict,
) -> None:
    """Run post-import verification — providers, paths, aliases, platform bindings.

    Populates report.needs_reauth and report.incompatible.
    """
    target_os = target_platform["os"]
    target_home = target_platform["home"]

    print()
    print(color("  Running post-import verification...", Colors.CYAN))

    # --- 1. Provider / auth check ---
    auth_issues = _verify_providers(target_home)
    for item in auth_issues:
        report.needs_reauth.append(item)

    # --- 2. Path existence check for remapped paths ---
    path_issues = _verify_remapped_paths(target_home)
    for item in path_issues:
        report.needs_reauth.append(item)

    # --- 3. External tool availability ---
    tool_issues = _verify_external_tools()
    for item in tool_issues:
        report.incompatible.append(item)

    # --- 4. Platform-specific binding warnings ---
    platform_issues = _verify_platform_bindings(target_os, target_home)
    for item in platform_issues:
        report.incompatible.append(item)

    # Print results of post-import check
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
    if auth_path.exists():
        try:
            auth_data = json.loads(auth_path.read_text(encoding="utf-8"))
        except Exception:
            pass

    for provider_name, provider_config in providers.items():
        api_key = provider_config.get("api_key", "")
        # Check if it's a placeholder/empty that needs filling
        if not api_key or api_key in ("", "your-api-key-here", "env:DASHSCOPE_API_KEY"):
            # Check if auth.json has it as env ref
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

    # Check terminal.cwd
    terminal = config.get("terminal", {})
    cwd = terminal.get("cwd", "")
    if cwd and not Path(cwd).exists():
        issues.append(f"working directory does not exist: {cwd}")

    # Check external_dirs
    for d in config.get("external_dirs", []):
        if d and not Path(d).exists():
            issues.append(f"external directory not found: {d}  (create or update config)")

    # Check docker volume host paths
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

    # Check each unique tool referenced in aliases
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


# ============================================================================
# Migration report printer
# ============================================================================


def _print_migration_report(
    report: MigrationReport,
    manifest: dict,
    target_platform: dict,
) -> None:
    """Print a categorized migration report."""
    source_os = manifest.get("source_os", "unknown")
    target_os = target_platform["os"]
    cross_platform = source_os != target_os and source_os != "unknown"

    # Header
    print(color("  Migration Report", Colors.CYAN))
    print(color("  ─────────────────", Colors.CYAN))

    # Summary counts
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

    # Cross-platform warning
    if cross_platform:
        print(color(f"  ⚠ Cross-platform migration: {source_os} → {target_os}", Colors.YELLOW))
        print()

    # Detailed sections
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


# ============================================================================
# Interactive mode
# ============================================================================


def _run_interactive(
    bundle_path: Path,
    source_home: Path,
    target_home: Path,
    manifest: dict,
    target_platform: dict,
    report: MigrationReport,
) -> None:
    """Guided interactive migration flow."""
    import getpass

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
    preset = manifest.get("preset", "safe")
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
        print(f"  Home directory paths will be remapped:")
        print(f"    {source_home} → {target_home}")
        print()

    # Step 5: Confirm
    print(color("  Step 4: Confirm", Colors.CYAN))
    print(color("  ─────────────────", Colors.CYAN))

    try:
        response = getpass.getpass("  Proceed with import? [y/N]  ").strip().lower()
    except KeyboardInterrupt:
        print(color("\n\nCancelled.", Colors.YELLOW))
        sys.exit(130)

    if response not in ("y", "yes"):
        print(color("\n  Import cancelled.", Colors.YELLOW))
        sys.exit(0)

    print()

    # Backup
    _backup_conflicts(bundle_path)

    # Extract
    print(color("  Extracting bundle...", Colors.CYAN))
    _extract_with_remap(bundle_path, source_home, target_home, report)

    _remap_config_paths(source_home, target_home, manifest)

    _post_import_verify(report, manifest, target_platform)

    print()
    print(color("✓ Migration complete!", Colors.GREEN))
    print()
    _print_migration_report(report, manifest, target_platform)


# ============================================================================
# Verify
# ============================================================================


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

        # Integrity check
        print(color("  Checking bundle integrity...", Colors.CYAN))
        try:
            with tarfile.open(bundle_path, "r:gz") as tf:
                for member in tf:
                    if member.size > 0:
                        pass
            results["passed"].append("Bundle integrity (CRC valid)")
        except Exception as e:
            results["failed"].append(f"Bundle integrity: {e}")

        # config.yaml check
        with tarfile.open(bundle_path, "r:gz") as tf:
            try:
                f = tf.extractfile("config.yaml")
                if f:
                    yaml.safe_load(f.read())
                    results["passed"].append("config.yaml valid YAML")
                else:
                    results["warnings"].append("config.yaml not in bundle")
            except Exception as e:
                results["failed"].append(f"config.yaml: {e}")

        # Skills structure
        skills_found = 0
        with tarfile.open(bundle_path, "r:gz") as tf:
            for member in tf.getmembers():
                if member.name.startswith("skills/") and member.name.endswith("/SKILL.md"):
                    skills_found += 1
        if skills_found > 0:
            results["passed"].append(f"Skills structure OK ({skills_found} skills)")
        else:
            results["warnings"].append("No skills found in bundle")

        # secrets check
        with tarfile.open(bundle_path, "r:gz") as tf:
            names = tf.getnames()
            has_env = "env" in names
            has_auth = "auth.json" in names
        if has_env or has_auth:
            results["passed"].append(f"Secrets included (env={has_env}, auth={has_auth})")
        else:
            results["warnings"].append("No secrets in bundle — API keys need re-entry after import")

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
                if auth_path.exists():
                    try:
                        auth_data = json.loads(auth_path.read_text(encoding="utf-8"))
                    except Exception:
                        pass
                missing = []
                for name, prov in providers.items():
                    if not prov.get("api_key"):
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


# ============================================================================
# Doctor
# ============================================================================


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
        total, used, free = sh.disk_usage(HERMES_HOME)
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
        except Exception:
            pass

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


# ============================================================================
# CLI entry point (integrated with main.py argparse)
# ============================================================================


def run_migrate(args):
    """Entry point called by main.py's cmd_migrate handler."""
    action = getattr(args, "action", None)

    if action is None:
        print("Run 'hermes migrate --help' to see available subcommands.")
        return

    interactive = getattr(args, "interactive", False)

    try:
        if action == "export":
            export_bundle(getattr(args, "output", None), getattr(args, "preset", "safe"))
        elif action == "import":
            import_bundle(
                getattr(args, "input", None),
                getattr(args, "preset", "safe"),
                getattr(args, "dry_run", False),
                interactive=interactive,
            )
        elif action == "verify":
            success = verify_bundle(getattr(args, "input", None))
            sys.exit(0 if success else 1)
        elif action == "doctor":
            success = run_doctor()
            sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print(color("\n\nCancelled.", Colors.YELLOW))
        sys.exit(130)
    except Exception as e:
        print(color(f"\n\nError: {e}", Colors.RED))
        sys.exit(1)


def main():
    """Parse sys.argv and dispatch (standalone invocation only)."""
    import argparse
    parser = argparse.ArgumentParser(
        "hermes migrate",
        description="Unified migration command for Hermes Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="action", help="Migration action")

    exp = subparsers.add_parser("export", help="Export Hermes to a migration bundle")
    exp.add_argument("--preset", "-p", choices=["safe", "full"], default="safe")
    exp.add_argument("--output", "-o")

    imp = subparsers.add_parser("import", help="Import from a migration bundle")
    imp.add_argument("--input", "-i", required=True)
    imp.add_argument("--preset", "-p", choices=["safe", "full"], default="safe")
    imp.add_argument("--dry-run", action="store_true")
    imp.add_argument("--interactive", action="store_true")

    ver = subparsers.add_parser("verify", help="Verify a bundle")
    ver.add_argument("--input", "-i")

    subparsers.add_parser("doctor", help="Check environment health")

    args = parser.parse_args()
    run_migrate(args)


if __name__ == "__main__":
    main()
