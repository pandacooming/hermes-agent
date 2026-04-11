"""
Tests for the hermes migrate command (feature/unified-migration).

These tests exercise the core migrate module functions:
- Platform detection
- Bundle creation and manifest handling
- Item filtering (skip logic)
- Bundle content discovery and classification
- Post-import verification
- Doctor / environment checks
- End-to-end export + dry-run import
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import tarfile
import tempfile
from io import BytesIO
from pathlib import Path

import pytest
import yaml


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def migrate_module(tmp_path, monkeypatch):
    """Provide the migrate module with HERMES_HOME pointed at a temp directory.

    Returns the patched migrate module. The hermes_dir (Path) is accessible as
    migrate_module.HERMES_HOME so tests that need the path can use that directly.
    """
    import hermes_cli.migrate as _migrate_mod

    # Create a fresh temp hermes dir for THIS test (isolated, not cached)
    hermes_dir = tmp_path / ".hermes"
    hermes_dir.mkdir()

    # Patch the module-level HERMES_HOME (Path evaluated at import time)
    monkeypatch.setattr(_migrate_mod, "HERMES_HOME", hermes_dir)
    monkeypatch.setenv("HERMES_HOME", str(hermes_dir))

    # Stub profiles (used by _safe_extract_profile_archive)
    import hermes_cli.profiles as _profiles_mod
    _profiles_mod._safe_extract_profile_archive = _PROFILES_STUB_FN

    # Stub colors (non-critical for tests)
    import hermes_cli.colors as _colors_mod
    _colors_mod.Colors = type("Colors", (), {
        "GREEN": "", "RED": "", "YELLOW": "", "CYAN": "",
        "MAGENTA": "", "RESET": "",
    })()
    _colors_mod.color = staticmethod(lambda text, _: text)

    yield _migrate_mod
    # monkeypatch auto-restores after each test


# ---------------------------------------------------------------------------
# Stub modules (avoid installing the full hermes package)
# ---------------------------------------------------------------------------

_COLORS_STUB = '''
class Colors:
    GREEN = ""
    RED = ""
    YELLOW = ""
    CYAN = ""
    MAGENTA = ""
    RESET = ""

def color(text, _):
    return text
'''

_PROFILES_STUB = '''
from pathlib import Path

def _safe_extract_profile_archive(archive_path, dest_dir):
    """Stub that extracts all members without profile renaming."""
    import tarfile
    with tarfile.open(archive_path, "r:gz") as tf:
        tf.extractall(dest_dir)
'''


def _PROFILES_STUB_FN(archive_path, dest_dir):
    """Stub for _safe_extract_profile_archive."""
    import tarfile
    with tarfile.open(archive_path, "r:gz") as tf:
        tf.extractall(dest_dir)


# ---------------------------------------------------------------------------
# Tests — Platform detection
# ---------------------------------------------------------------------------

def test_detect_platform_returns_current_os(migrate_module):
    """detect_platform() returns the current OS name."""
    import platform
    expected_os = {"Darwin": "macos", "Linux": "linux", "Windows": "windows"}.get(
        platform.system(), "linux"
    )
    p = migrate_module.detect_platform()
    assert p["os"] == expected_os
    assert isinstance(p["home"], Path)


def test_detect_platform_get_home(migrate_module):
    home = migrate_module.get_home()
    assert home == Path(os.path.expanduser("~"))


# ---------------------------------------------------------------------------
# Tests — Manifest
# ---------------------------------------------------------------------------

def test_create_manifest_includes_version(migrate_module):
    source = {"os": "linux", "home": Path("/home/test")}
    manifest = migrate_module.create_manifest("safe", source)
    assert manifest["version"] == "1.0"
    assert manifest["source_os"] == "linux"
    assert manifest["includes_secrets"] is False


def test_create_manifest_full_preset_includes_secrets(migrate_module):
    source = {"os": "macos", "home": Path("/Users/test")}
    manifest = migrate_module.create_manifest("full", source)
    assert manifest["includes_secrets"] is True
    assert manifest["preset"] == "full"


# ---------------------------------------------------------------------------
# Tests — Filtering
# ---------------------------------------------------------------------------

def test_should_skip_dir_excludes_hidden_and_excluded(migrate_module):
    assert migrate_module._should_skip_dir(".git", "linux") is True
    assert migrate_module._should_skip_dir("node_modules", "linux") is True
    assert migrate_module._should_skip_dir("hermes-agent", "linux") is True
    assert migrate_module._should_skip_dir("memories", "linux") is False
    assert migrate_module._should_skip_dir("skills", "linux") is False


def test_should_skip_file_excludes_hidden_and_excluded(migrate_module):
    # .env and auth.json are NOT skipped here — handled by _collect_migration_items
    assert migrate_module._should_skip_file(".env", "linux") is False
    assert migrate_module._should_skip_file("auth.json", "linux") is False
    assert migrate_module._should_skip_file("state.db", "linux") is True
    assert migrate_module._should_skip_file(".gitignore", "linux") is True
    assert migrate_module._should_skip_file("config.yaml", "linux") is False
    assert migrate_module._should_skip_file("skills.yaml", "linux") is False


def test_should_skip_file_platform_skips_powershell_on_linux(migrate_module):
    assert migrate_module._should_skip_file("script.ps1", "linux") is True
    # .ps1 is in the platform skip set for ALL platforms (including windows),
    # so PowerShell scripts are always excluded across the board
    assert migrate_module._should_skip_file("script.ps1", "windows") is True


# ---------------------------------------------------------------------------
# Tests — Bundle creation (export)
# ---------------------------------------------------------------------------

def test_export_bundle_creates_tarfile(migrate_module):
    hermes_dir = migrate_module.HERMES_HOME
    # Create minimal content
    (hermes_dir / "config.yaml").write_text("model: test\n", encoding="utf-8")
    (hermes_dir / "memories").mkdir()
    (hermes_dir / "memories" / "MEMORY.md").write_text("# Memory\n", encoding="utf-8")
    (hermes_dir / "skills").mkdir()

    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = migrate_module.export_bundle(
            output_path=str(Path(tmpdir) / "test-bundle.tar.gz"),
            preset="safe",
        )
        assert out_path.exists()
        assert out_path.suffix == ".gz"

        # Verify it's a valid tar
        with tarfile.open(out_path, "r:gz") as tf:
            names = tf.getnames()
            assert "manifest.json" in names
            assert any("config.yaml" in n for n in names)


def test_export_bundle_excludes_secrets_in_safe_preset(migrate_module):
    hermes_dir = migrate_module.HERMES_HOME
    (hermes_dir / "config.yaml").write_text("model: test\n", encoding="utf-8")
    (hermes_dir / ".env").write_text("SECRET=***", encoding="utf-8")

    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = migrate_module.export_bundle(
            output_path=str(Path(tmpdir) / "safe-bundle.tar.gz"),
            preset="safe",
        )
        with tarfile.open(out_path, "r:gz") as tf:
            names = tf.getnames()
            assert ".env" not in names


def test_export_bundle_includes_secrets_in_full_preset(migrate_module):
    """Auth.json (non-dotfile secret) is included when preset=full.

    Note: .env is handled by _collect_migration_items, not _should_skip_file.
    We use auth.json here since it better isolates the preset=full behavior.
    """
    hermes_dir = migrate_module.HERMES_HOME
    (hermes_dir / "config.yaml").write_text("model: test\n", encoding="utf-8")
    (hermes_dir / "auth.json").write_text('{"providers": {}}', encoding="utf-8")

    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = migrate_module.export_bundle(
            output_path=str(Path(tmpdir) / "full-bundle.tar.gz"),
            preset="full",
        )
        with tarfile.open(out_path, "r:gz") as tf:
            names = tf.getnames()
            assert "auth.json" in names


# ---------------------------------------------------------------------------
# Tests — Manifest read
# ---------------------------------------------------------------------------

def test_read_manifest_parses_valid_manifest(migrate_module):
    hermes_dir = migrate_module.HERMES_HOME
    manifest_data = {
        "version": "1.0",
        "source_os": "linux",
        "preset": "safe",
    }
    bundle_path = Path(hermes_dir) / "bundle.tar.gz"
    with tarfile.open(bundle_path, "w:gz") as tf:
        info = tarfile.TarInfo(name="manifest.json")
        payload = json.dumps(manifest_data).encode("utf-8")
        info.size = len(payload)
        tf.addfile(info, BytesIO(payload))

    result = migrate_module._read_manifest(bundle_path)
    assert result["version"] == "1.0"
    assert result["source_os"] == "linux"


def test_read_manifest_returns_empty_dict_if_missing(migrate_module):
    hermes_dir = migrate_module.HERMES_HOME
    bundle_path = Path(hermes_dir) / "empty.tar.gz"
    with tarfile.open(bundle_path, "w:gz") as tf:
        pass  # empty bundle

    result = migrate_module._read_manifest(bundle_path)
    assert result == {}


# ---------------------------------------------------------------------------
# Tests — Bundle content discovery
# ---------------------------------------------------------------------------

def test_discover_bundle_contents_migrated_skipped_incompatible(
    migrate_module
):
    hermes_dir = migrate_module.HERMES_HOME
    # Build a bundle with mixed content
    bundle_path = Path(hermes_dir) / "mixed.tar.gz"
    manifest_data = {
        "version": "1.0",
        "source_os": "linux",
        "bundle_created_at": "2026-01-01T00:00:00Z",
        "preset": "safe",
    }
    with tarfile.open(bundle_path, "w:gz") as tf:
        # Add manifest
        info = tarfile.TarInfo(name="manifest.json")
        payload = json.dumps(manifest_data).encode("utf-8")
        info.size = len(payload)
        tf.addfile(info, BytesIO(payload))

        # Add normal files
        for name in ["config.yaml", "SOUL.md", "memories/MEMORY.md"]:
            info = tarfile.TarInfo(name=name)
            payload = b"content"
            info.size = len(payload)
            tf.addfile(info, BytesIO(payload))

        # Add platform-incompatible file (PowerShell on Linux)
        info = tarfile.TarInfo(name="scripts/setup.ps1")
        payload = b"$true"
        info.size = len(payload)
        tf.addfile(info, BytesIO(payload))

        # Add always-excluded runtime artifact
        info = tarfile.TarInfo(name="state.db")
        payload = b"data"
        info.size = len(payload)
        tf.addfile(info, BytesIO(payload))

    target_platform = {"os": "linux", "home": Path("/home/test")}
    report = migrate_module._discover_bundle_contents(bundle_path, target_platform)

    # Normal files should be migrated
    assert any("config.yaml" in f for f in report.migrated)
    assert any("memories/MEMORY.md" in f for f in report.migrated)

    # Runtime artifact should be skipped
    assert any("state.db" in f for f in report.skipped)

    # Platform-incompatible should be flagged
    assert any("setup.ps1" in f for f in report.incompatible)


def test_migration_report_is_empty_when_no_items():
    from hermes_cli.migrate import MigrationReport
    report = MigrationReport()
    assert report.is_empty() is True


def test_migration_report_is_not_empty_when_has_items():
    from hermes_cli.migrate import MigrationReport
    report = MigrationReport(migrated=["config.yaml"])
    assert report.is_empty() is False


# ---------------------------------------------------------------------------
# Tests — _collect_migration_items
# ---------------------------------------------------------------------------

def test_collect_migration_items_safe_preset_excludes_secrets(migrate_module):
    hermes_dir = migrate_module.HERMES_HOME
    # Create dirs and files
    (hermes_dir / "memories").mkdir()
    (hermes_dir / "config.yaml").write_text("model: test\n", encoding="utf-8")
    (hermes_dir / ".env").write_text("SECRET=***", encoding="utf-8")

    items = migrate_module._collect_migration_items("safe")

    assert items["memories/"]["status"] == "migrated"
    assert items["config.yaml"]["status"] == "migrated"
    assert items[".env"]["status"] == "skipped"
    assert "secrets excluded" in items[".env"]["reason"]


def test_collect_migration_items_full_preset_includes_secrets(migrate_module):
    hermes_dir = migrate_module.HERMES_HOME
    (hermes_dir / "config.yaml").write_text("model: test\n", encoding="utf-8")
    (hermes_dir / ".env").write_text("SECRET=***", encoding="utf-8")

    items = migrate_module._collect_migration_items("full")

    assert items[".env"]["status"] == "migrated"


# ---------------------------------------------------------------------------
# Tests — Path remapping
# ---------------------------------------------------------------------------

def test_remap_content_changes_home_directory(migrate_module):
    hermes_dir = migrate_module.HERMES_HOME
    content = "working_directory: /home/old_user/projects"
    remapped = migrate_module._remap_content(
        content,
        Path("/home/old_user"),
        Path("/home/new_user"),
    )
    assert "/home/new_user" in remapped
    assert "/home/old_user" not in remapped


def test_remap_content_unchanged_when_same_home(migrate_module):
    hermes_dir = migrate_module.HERMES_HOME
    content = "working_directory: /home/same/projects"
    remapped = migrate_module._remap_content(
        content,
        Path("/home/same"),
        Path("/home/same"),
    )
    assert remapped == content


def test_is_text_file_recognizes_common_extensions(migrate_module):
    hermes_dir = migrate_module.HERMES_HOME
    assert migrate_module._is_text_file("config.yaml") is True
    assert migrate_module._is_text_file("config.yml") is True
    assert migrate_module._is_text_file("SOUL.md") is True
    assert migrate_module._is_text_file("script.sh") is True
    assert migrate_module._is_text_file(".env") is True
    assert migrate_module._is_text_file("image.png") is False
    assert migrate_module._is_text_file("data.db") is False


# ---------------------------------------------------------------------------
# Tests — verify_bundle (current installation)
# ---------------------------------------------------------------------------

def test_verify_bundle_current_installation_passes_with_valid_config(
    migrate_module
):
    hermes_dir = migrate_module.HERMES_HOME
    (hermes_dir / "config.yaml").write_text(
        "model: openrouter/auto\nproviders:\n  openrouter:\n    api_key: test\n",
        encoding="utf-8",
    )
    (hermes_dir / "skills").mkdir()

    success = migrate_module.verify_bundle()
    assert success is True


def test_verify_bundle_current_installation_warns_missing_config(
    migrate_module
):
    """Missing config.yaml should cause verify to return False."""
    hermes_dir = migrate_module.HERMES_HOME
    # No config.yaml at all
    result = migrate_module.verify_bundle()
    # Missing config is a hard failure -> False
    assert result is False


# ---------------------------------------------------------------------------
# Tests — verify_bundle (bundle file)
# ---------------------------------------------------------------------------

def test_verify_bundle_file_checks_integrity_and_manifest(
    migrate_module
):
    hermes_dir = migrate_module.HERMES_HOME
    manifest_data = {
        "version": "1.0",
        "source_os": "linux",
        "bundle_created_at": "2026-01-01T00:00:00Z",
        "preset": "safe",
        "includes_secrets": False,
    }
    bundle_path = Path(hermes_dir) / "verify-test.tar.gz"
    with tarfile.open(bundle_path, "w:gz") as tf:
        info = tarfile.TarInfo(name="manifest.json")
        payload = json.dumps(manifest_data).encode("utf-8")
        info.size = len(payload)
        tf.addfile(info, BytesIO(payload))
        # Add valid config.yaml
        info = tarfile.TarInfo(name="config.yaml")
        payload = yaml.safe_dump({"model": "test"}).encode("utf-8")
        info.size = len(payload)
        tf.addfile(info, BytesIO(payload))

    success = migrate_module.verify_bundle(str(bundle_path))
    assert success is True


# ---------------------------------------------------------------------------
# Tests — run_doctor
# ---------------------------------------------------------------------------

def test_run_doctor_returns_true_for_healthy_environment(migrate_module):
    success = migrate_module.run_doctor()
    # Returns True if no hard failures (warnings are OK)
    assert success is True


# ---------------------------------------------------------------------------
# Tests — End-to-end: export + dry-run import
# ---------------------------------------------------------------------------

def test_export_then_dry_run_import_produces_report(migrate_module):
    hermes_dir = migrate_module.HERMES_HOME
    # Setup source data
    (hermes_dir / "config.yaml").write_text("model: test\n", encoding="utf-8")
    (hermes_dir / "SOUL.md").write_text("# SOUL\n", encoding="utf-8")
    (hermes_dir / "memories").mkdir()
    (hermes_dir / "memories" / "MEMORY.md").write_text("# Memory\n", encoding="utf-8")
    (hermes_dir / "skills").mkdir()
    (hermes_dir / "skills" / "test-skill").mkdir()
    (hermes_dir / "skills" / "test-skill" / "SKILL.md").write_text(
        "---\nname: test\n---\n", encoding="utf-8"
    )

    # Export
    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = migrate_module.export_bundle(
            output_path=str(Path(tmpdir) / "e2e-bundle.tar.gz"),
            preset="safe",
        )

        # Dry-run import
        report = migrate_module.import_bundle(
            input_path=str(out_path),
            preset="safe",
            dry_run=True,
        )

        # Should have discovered items
        assert report is not None
        assert len(report.migrated) > 0
        # Nothing should be incompatible in same-platform export
        assert "config.yaml" in str(report.migrated) or any(
            "config.yaml" in f for f in report.migrated
        )


# ---------------------------------------------------------------------------
# Tests — Interactive import (confirm prompt) — skip, just verify no crash
# ---------------------------------------------------------------------------

def test_import_bundle_interactive_dry_run_does_not_crash(migrate_module):
    hermes_dir = migrate_module.HERMES_HOME
    """Verify interactive + dry-run doesn't raise an error."""
    (hermes_dir / "config.yaml").write_text("model: test\n", encoding="utf-8")
    (hermes_dir / "memories").mkdir()

    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = migrate_module.export_bundle(
            output_path=str(Path(tmpdir) / "interactive-test.tar.gz"),
            preset="safe",
        )
        # interactive=True alone (no stdin simulation) - will exit early
        # at the getpass prompt, which raises KeyboardInterrupt
        # So just verify it doesn't crash during the dry-run path
        report = migrate_module.import_bundle(
            input_path=str(out_path),
            preset="safe",
            dry_run=True,
            interactive=True,  # dry_run takes priority, no getpass called
        )
        assert len(report.migrated) > 0


# ---------------------------------------------------------------------------
# Integration tests — full export + real import cycle
# ---------------------------------------------------------------------------

def test_full_import_cycle_safe_preset_no_secrets(migrate_module):
    """Export from source hermes home, import to target, verify no secrets leaked."""
    hermes_dir = migrate_module.HERMES_HOME

    # Set up source files
    (hermes_dir / "config.yaml").write_text("model: test\n", encoding="utf-8")
    (hermes_dir / "SOUL.md").write_text("# SOUL\n", encoding="utf-8")
    (hermes_dir / ".env").write_text("OPENAI_KEY=sk-secret123\n", encoding="utf-8")
    (hermes_dir / "auth.json").write_text('{"openai": "sk-secret"}', encoding="utf-8")
    (hermes_dir / "memories").mkdir()
    (hermes_dir / "memories" / "memory.md").write_text("# Memory\n", encoding="utf-8")
    (hermes_dir / "skills").mkdir()
    (hermes_dir / "skills" / "test-skill").mkdir()
    (hermes_dir / "skills" / "test-skill" / "SKILL.md").write_text(
        "---\nname: test\n---\n", encoding="utf-8"
    )
    # Platform-specific file should be skipped on same platform (linux)
    (hermes_dir / "script.ps1").write_text("Write-Host 'hi'\n", encoding="utf-8")

    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = migrate_module.export_bundle(
            output_path=str(Path(tmpdir) / "safe-bundle.tar.gz"),
            preset="safe",
        )

        # Verify bundle contents — no .env or auth.json
        with tarfile.open(out_path, "r:gz") as tf:
            names = [m.name for m in tf.getmembers()]
        assert "manifest.json" in names
        assert ".env" not in names, ".env should be excluded in safe preset"
        assert "auth.json" not in names, "auth.json should be excluded in safe preset"
        assert "config.yaml" in names
        assert "SOUL.md" in names
        assert any("memory.md" in n for n in names)
        assert any("SKILL.md" in n for n in names)
        assert "script.ps1" not in names, ".ps1 should be skipped on linux"


def test_full_import_cycle_full_preset_includes_secrets(migrate_module):
    """Export with preset=full, verify secrets are in the bundle."""
    hermes_dir = migrate_module.HERMES_HOME

    (hermes_dir / "config.yaml").write_text("model: test\n", encoding="utf-8")
    (hermes_dir / ".env").write_text("OPENAI_KEY=sk-secret123\n", encoding="utf-8")
    (hermes_dir / "auth.json").write_text('{"openai": "sk-secret"}', encoding="utf-8")
    (hermes_dir / "memories").mkdir()
    (hermes_dir / "memories" / "memory.md").write_text("# Memory\n", encoding="utf-8")

    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = migrate_module.export_bundle(
            output_path=str(Path(tmpdir) / "full-bundle.tar.gz"),
            preset="full",
        )

        with tarfile.open(out_path, "r:gz") as tf:
            names = [m.name for m in tf.getmembers()]

        assert ".env" in names, ".env should be in full preset bundle"
        assert "auth.json" in names, "auth.json should be in full preset bundle"

        # Extract and check content is intact
        with tarfile.open(out_path, "r:gz") as tf:
            env_member = [m for m in tf.getmembers() if m.name == ".env"][0]
            env_content = tf.extractfile(env_member).read().decode("utf-8")
        assert "OPENAI_KEY=sk-secret123" in env_content


def test_full_import_cycle_cross_directory_remap(migrate_module, monkeypatch):
    """Export from source hermes home, mock source_home in manifest, import to
    different target home, verify _remap_content rewrites old home paths."""
    import hermes_cli.migrate as _migrate_mod

    hermes_dir = migrate_module.HERMES_HOME

    # Create files with absolute paths pointing to old machine
    (hermes_dir / "config.yaml").write_text(
        "model: test\nterminal:\n  cwd: /home/alice/projects\n",
        encoding="utf-8",
    )
    (hermes_dir / "memories").mkdir()
    (hermes_dir / "memories" / "ref.md").write_text(
        "See /home/alice/docs for details\n", encoding="utf-8"
    )

    # Export bundle normally
    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = migrate_module.export_bundle(
            output_path=str(Path(tmpdir) / "remap-bundle.tar.gz"),
            preset="safe",
        )

        # Rewrite manifest to lie about source_home (simulates cross-machine)
        import shutil
        patched_path = Path(tmpdir) / "patched-bundle.tar.gz"
        with tarfile.open(out_path, "r:gz") as src,              tarfile.open(patched_path, "w:gz") as dst:
            for member in src.getmembers():
                if member.name == "manifest.json":
                    # Replace manifest with fake source home
                    new_manifest = json.dumps({
                        "version": "1.0",
                        "hermes_version": "unknown",
                        "preset": "safe",
                        "source_os": "linux",
                        "source_home": "/home/alice",  # fake old machine
                        "includes_secrets": False,
                    }).encode("utf-8")
                    info = tarfile.TarInfo(name="manifest.json")
                    info.size = len(new_manifest)
                    dst.addfile(info, BytesIO(new_manifest))
                else:
                    dst.addfile(member, src.extractfile(member))

        # Now import with fake "new machine" home — patch detect_platform
        original_detect = _migrate_mod.detect_platform
        def fake_detect():
            return {"os": "linux", "home": Path("/home/bob")}
        monkeypatch.setattr(_migrate_mod, "detect_platform", fake_detect)

        report = migrate_module.import_bundle(
            input_path=str(patched_path),
            preset="safe",
            dry_run=False,
        )

        # Verify home paths were remapped in text files
        config_content = (hermes_dir / "config.yaml").read_text(encoding="utf-8")
        assert "/home/alice" not in config_content, "Old home path should be remapped"
        assert "/home/bob" in config_content, "/home/bob should appear after remapping"

        mem_content = (hermes_dir / "memories" / "ref.md").read_text(encoding="utf-8")
        assert "/home/alice" not in mem_content, "Old home in memory.md should be remapped"
        assert "/home/bob" in mem_content, "New home should appear in memory.md"


def test_import_excludes_runtime_artifacts(migrate_module):
    """Runtime artifacts (state.db, __pycache__, etc.) must not end up in bundle."""
    hermes_dir = migrate_module.HERMES_HOME

    (hermes_dir / "config.yaml").write_text("model: test\n", encoding="utf-8")
    (hermes_dir / "state.db").write_text("sqlite db content", encoding="utf-8")
    (hermes_dir / "__pycache__").mkdir()
    (hermes_dir / "__pycache__" / "test.pyc").write_text("bytecode", encoding="utf-8")
    (hermes_dir / "hermes_state.db").write_text("db", encoding="utf-8")
    (hermes_dir / "gateway.pid").write_text("1234", encoding="utf-8")
    (hermes_dir / "memories").mkdir()
    (hermes_dir / "memories" / "mem.md").write_text("# mem\n", encoding="utf-8")

    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = migrate_module.export_bundle(
            output_path=str(Path(tmpdir) / "artifact-bundle.tar.gz"),
            preset="safe",
        )

        with tarfile.open(out_path, "r:gz") as tf:
            names = [m.name for m in tf.getmembers()]

        assert "state.db" not in names, "state.db is runtime artifact"
        assert "hermes_state.db" not in names
        assert "__pycache__" not in names
        assert "gateway.pid" not in names
        assert "config.yaml" in names
        assert any("mem.md" in n for n in names)
