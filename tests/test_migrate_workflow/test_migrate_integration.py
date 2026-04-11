"""
Integration tests for hermes migrate — real-filesystem, no mocking.

Uses real temp directories with HERMES_HOME pointed at them. Import tests
use subprocess CLI calls to truly isolate the destination environment.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
from io import BytesIO
from pathlib import Path

import pytest


# --------------------------------------------------------------------------$
# Fixtures — module-level HERMES_HOME management
# --------------------------------------------------------------------------$


@pytest.fixture(autouse=True)
def _set_hermes_home(tmp_path):
    """
    Auto-runs for every test. Sets HERMES_HOME + module var to a fresh
    temp directory. No cleanup needed — tmp_path is auto-deleted.

    Uses a unique dir per test (pytest's tmp_path is already per-test).
    """
    hermes_dir = tmp_path / ".hermes"
    hermes_dir.mkdir()

    # Set env var
    old_hermes_home = os.environ.get("HERMES_HOME")
    os.environ["HERMES_HOME"] = str(hermes_dir)

    # Patch the module-level HERMES_HOME so direct function calls use the temp dir
    import hermes_cli.migrate as _m_mod
    _original_hermes_home = _m_mod.HERMES_HOME
    _m_mod.HERMES_HOME = hermes_dir

    # Also patch hermes_cli.profiles if it exists (used during profile archive extraction)
    try:
        import hermes_cli.profiles as _p_mod
        if hasattr(_p_mod, "HERMES_HOME"):
            _orig_profiles_home = _p_mod.HERMES_HOME
            _p_mod.HERMES_HOME = hermes_dir
        else:
            _orig_profiles_home = None
    except ImportError:
        _orig_profiles_home = None

    yield hermes_dir

    # Restore
    if old_hermes_home is not None:
        os.environ["HERMES_HOME"] = old_hermes_home
    else:
        os.environ.pop("HERMES_HOME", None)
    _m_mod.HERMES_HOME = _original_hermes_home
    if _orig_profiles_home is not None:
        _p_mod.HERMES_HOME = _orig_profiles_home


# --------------------------------------------------------------------------$
# Integration tests — bundle export
# --------------------------------------------------------------------------$


class TestExportBundleIntegrity:
    """Verify export produces valid, well-formed bundles."""

    def test_export_creates_valid_tarball(self, tmp_path):
        """.tar.gz is readable and contains manifest + config + memories."""
        from hermes_cli.migrate import export_bundle

        hermes = tmp_path / ".hermes"
        (hermes / "config.yaml").write_text("model: test\n", encoding="utf-8")
        (hermes / "SOUL.md").write_text("# SOUL\n", encoding="utf-8")
        (hermes / "memories").mkdir()
        (hermes / "memories" / "test.md").write_text("# Memory\n", encoding="utf-8")

        with tempfile.TemporaryDirectory() as out_dir:
            out = export_bundle(output_path=str(Path(out_dir) / "bundle.tar.gz"), preset="safe")
            assert out.exists()

            with tarfile.open(out, "r:gz") as tf:
                names = tf.getnames()
            assert "manifest.json" in names
            assert "config.yaml" in names
            assert "SOUL.md" in names

    def test_export_safe_preset_excludes_dotenv(self, tmp_path):
        """safe preset: .env is NOT in the bundle."""
        from hermes_cli.migrate import export_bundle

        hermes = tmp_path / ".hermes"
        (hermes / "config.yaml").write_text("model: test\n", encoding="utf-8")
        (hermes / ".env").write_text("OPENAI_KEY=sk-123\n", encoding="utf-8")

        with tempfile.TemporaryDirectory() as out_dir:
            out = export_bundle(output_path=str(Path(out_dir) / "safe.tar.gz"), preset="safe")
            with tarfile.open(out, "r:gz") as tf:
                names = tf.getnames()
            assert ".env" not in names

    def test_export_full_preset_includes_dotenv(self, tmp_path):
        """full preset: .env IS in the bundle and content is intact."""
        from hermes_cli.migrate import export_bundle

        hermes = tmp_path / ".hermes"
        (hermes / "config.yaml").write_text("model: test\n", encoding="utf-8")
        (hermes / ".env").write_text("OPENAI_KEY=sk-123\n", encoding="utf-8")

        with tempfile.TemporaryDirectory() as out_dir:
            out = export_bundle(output_path=str(Path(out_dir) / "full.tar.gz"), preset="full")
            with tarfile.open(out, "r:gz") as tf:
                names = tf.getnames()
            assert ".env" in names
            with tarfile.open(out, "r:gz") as tf:
                content = tf.extractfile(".env").read()
            assert b"OPENAI_KEY=sk-123" in content

    def test_export_full_preset_includes_auth_json(self, tmp_path):
        """full preset: auth.json is in the bundle."""
        from hermes_cli.migrate import export_bundle

        hermes = tmp_path / ".hermes"
        (hermes / "config.yaml").write_text("model: test\n", encoding="utf-8")
        (hermes / "auth.json").write_text('{"openai": "sk-xxx"}', encoding="utf-8")

        with tempfile.TemporaryDirectory() as out_dir:
            out = export_bundle(output_path=str(Path(out_dir) / "full.tar.gz"), preset="full")
            with tarfile.open(out, "r:gz") as tf:
                names = tf.getnames()
            assert "auth.json" in names

    def test_export_safe_preset_excludes_auth_json(self, tmp_path):
        """safe preset: auth.json is NOT in the bundle."""
        from hermes_cli.migrate import export_bundle

        hermes = tmp_path / ".hermes"
        (hermes / "config.yaml").write_text("model: test\n", encoding="utf-8")
        (hermes / "auth.json").write_text('{"openai": "sk-xxx"}', encoding="utf-8")

        with tempfile.TemporaryDirectory() as out_dir:
            out = export_bundle(output_path=str(Path(out_dir) / "safe.tar.gz"), preset="safe")
            with tarfile.open(out, "r:gz") as tf:
                names = tf.getnames()
            assert "auth.json" not in names

    def test_export_nested_files_in_memories(self, tmp_path):
        """memories/ subdirectories (including deeply nested) are included."""
        from hermes_cli.migrate import export_bundle

        hermes = tmp_path / ".hermes"
        (hermes / "config.yaml").write_text("model: test\n", encoding="utf-8")
        memories = hermes / "memories"
        memories.mkdir()
        (memories / "alpha.md").write_text("# Alpha\n", encoding="utf-8")
        subdir = memories / "sub"
        subdir.mkdir()
        (subdir / "gamma.md").write_text("# Gamma\n", encoding="utf-8")

        with tempfile.TemporaryDirectory() as out_dir:
            out = export_bundle(output_path=str(Path(out_dir) / "bundle.tar.gz"), preset="safe")
            with tarfile.open(out, "r:gz") as tf:
                names = tf.getnames()
            assert any("alpha.md" in n for n in names)
            assert any("gamma.md" in n for n in names)

    def test_export_nested_files_in_skills(self, tmp_path):
        """skills/ with nested SKILL.md and reference files are included."""
        from hermes_cli.migrate import export_bundle

        hermes = tmp_path / ".hermes"
        (hermes / "config.yaml").write_text("model: test\n", encoding="utf-8")
        skills = hermes / "skills"
        skills.mkdir()
        skill_a = skills / "skill-a"
        skill_a.mkdir()
        (skill_a / "SKILL.md").write_text("---\nname: skill-a\n---\n", encoding="utf-8")
        (skill_a / "refs").mkdir()
        (skill_a / "refs" / "api.md").write_text("# API\n", encoding="utf-8")

        with tempfile.TemporaryDirectory() as out_dir:
            out = export_bundle(output_path=str(Path(out_dir) / "bundle.tar.gz"), preset="safe")
            with tarfile.open(out, "r:gz") as tf:
                names = tf.getnames()
            assert any("skill-a/SKILL.md" in n for n in names)
            assert any("api.md" in n for n in names)

    def test_export_excludes_runtime_artifacts(self, tmp_path):
        """state.db, hermes_state.db, __pycache__, .pyc, gateway.pid are never bundled."""
        from hermes_cli.migrate import export_bundle

        hermes = tmp_path / ".hermes"
        (hermes / "config.yaml").write_text("model: test\n", encoding="utf-8")
        (hermes / "state.db").write_text("db", encoding="utf-8")
        (hermes / "hermes_state.db").write_text("db", encoding="utf-8")
        (hermes / "__pycache__").mkdir()
        (hermes / "__pycache__" / "test.cpython-311.pyc").write_text("byte", encoding="utf-8")
        (hermes / "gateway.pid").write_text("123", encoding="utf-8")

        with tempfile.TemporaryDirectory() as out_dir:
            out = export_bundle(output_path=str(Path(out_dir) / "bundle.tar.gz"), preset="safe")
            with tarfile.open(out, "r:gz") as tf:
                names = tf.getnames()
            assert "state.db" not in names
            assert "hermes_state.db" not in names
            assert "__pycache__" not in names
            assert "gateway.pid" not in names
            assert "config.yaml" in names

    def test_export_excludes_dotfiles(self, tmp_path):
        """.gitignore, .DS_Store are excluded; .env handled by preset logic."""
        from hermes_cli.migrate import export_bundle

        hermes = tmp_path / ".hermes"
        (hermes / "config.yaml").write_text("model: test\n", encoding="utf-8")
        (hermes / ".gitignore").write_text("*.pyc\n", encoding="utf-8")
        (hermes / ".DS_Store").write_text("blob", encoding="utf-8")
        (hermes / ".env").write_text("KEY=val\n", encoding="utf-8")  # excluded by safe

        with tempfile.TemporaryDirectory() as out_dir:
            out = export_bundle(output_path=str(Path(out_dir) / "bundle.tar.gz"), preset="safe")
            with tarfile.open(out, "r:gz") as tf:
                names = tf.getnames()
            assert ".gitignore" not in names
            assert ".DS_Store" not in names
            assert ".env" not in names

    def test_export_includes_sessions(self, tmp_path):
        """sessions/ is included in the bundle."""
        from hermes_cli.migrate import export_bundle

        hermes = tmp_path / ".hermes"
        (hermes / "config.yaml").write_text("model: test\n", encoding="utf-8")
        sessions = hermes / "sessions"
        sessions.mkdir()
        (sessions / "session-abc.json").write_text('{"id": "abc"}', encoding="utf-8")

        with tempfile.TemporaryDirectory() as out_dir:
            out = export_bundle(output_path=str(Path(out_dir) / "bundle.tar.gz"), preset="safe")
            with tarfile.open(out, "r:gz") as tf:
                names = tf.getnames()
            assert "sessions/session-abc.json" in names

    def test_export_includes_profiles(self, tmp_path):
        """profiles/ with sub-directory config is included."""
        from hermes_cli.migrate import export_bundle

        hermes = tmp_path / ".hermes"
        (hermes / "config.yaml").write_text("model: test\n", encoding="utf-8")
        profiles = hermes / "profiles"
        profiles.mkdir()
        (profiles / "coder").mkdir()
        (profiles / "coder" / "config.yaml").write_text("model: coder\n", encoding="utf-8")

        with tempfile.TemporaryDirectory() as out_dir:
            out = export_bundle(output_path=str(Path(out_dir) / "bundle.tar.gz"), preset="safe")
            with tarfile.open(out, "r:gz") as tf:
                names = tf.getnames()
            assert "profiles/coder/config.yaml" in names

    def test_export_manifest_metadata(self, tmp_path):
        """manifest has correct version, preset, source_os, includes_secrets, bundle_created_at."""
        from hermes_cli.migrate import export_bundle

        hermes = tmp_path / ".hermes"
        (hermes / "config.yaml").write_text("model: test\n", encoding="utf-8")

        with tempfile.TemporaryDirectory() as out_dir:
            out = export_bundle(output_path=str(Path(out_dir) / "bundle.tar.gz"), preset="safe")
            with tarfile.open(out, "r:gz") as tf:
                manifest = json.loads(tf.extractfile("manifest.json").read())

        assert manifest["version"] == "1.0"
        assert manifest["preset"] == "safe"
        assert manifest["includes_secrets"] is False
        assert manifest["source_os"] in ("linux", "darwin", "windows", "wsl")
        assert "bundle_created_at" in manifest


# --------------------------------------------------------------------------$
# Integration tests — bundle import
# --------------------------------------------------------------------------$


def _run_import_cli(bundle_path: str, preset: str, hermes_home: str, dry_run: bool = False) -> subprocess.CompletedProcess:
    """Run hermes migrate import as a subprocess with HERMES_HOME set."""
    cmd = [sys.executable, "-m", "hermes_cli.migrate", "import",
           "--input", bundle_path, "--preset", preset]
    if dry_run:
        cmd.append("--dry-run")
    env = {**os.environ, "HERMES_HOME": hermes_home}
    result = subprocess.run(cmd, capture_output=True, text=True, env=env)
    return result


class TestImportBundleContents:
    """Verify import extracts the right files to the right places."""

    def test_import_extracts_files(self, tmp_path):
        """After import, config.yaml, SOUL.md, memories/ are in hermes home."""
        from hermes_cli.migrate import export_bundle

        hermes = tmp_path / ".hermes"
        (hermes / "config.yaml").write_text("model: test\n", encoding="utf-8")
        (hermes / "SOUL.md").write_text("# SOUL\n", encoding="utf-8")
        memories = hermes / "memories"
        memories.mkdir()
        (memories / "note.md").write_text("# Note\n", encoding="utf-8")

        with tempfile.TemporaryDirectory() as out_dir:
            out = Path(out_dir) / "bundle.tar.gz"
            export_bundle(output_path=str(out), preset="safe")

            # Run import as CLI subprocess (true isolation)
            result = _run_import_cli(str(out), "safe", str(hermes))
            assert result.returncode == 0, f"import failed: {result.stderr}"

        assert (hermes / "config.yaml").exists()
        assert (hermes / "SOUL.md").exists()
        assert (hermes / "memories" / "note.md").exists()
        assert (hermes / "config.yaml").read_text(encoding="utf-8") == "model: test\n"

    def test_import_safe_preset_strips_dotenv(self, tmp_path):
        """safe preset import: .env is not extracted even if bundle contains it."""
        hermes = tmp_path / ".hermes"

        # Manually create a bundle with .env included
        with tempfile.TemporaryDirectory() as out_dir:
            bundle_path = Path(out_dir) / "malicious.tar.gz"
            with tarfile.open(bundle_path, "w:gz") as tf:
                manifest = json.dumps({
                    "version": "1.0", "preset": "safe", "source_os": "linux",
                    "source_home": "/home/alice", "includes_secrets": False,
                    "bundle_created_at": "2026-01-01T00:00:00Z",
                }).encode("utf-8")
                m = tarfile.TarInfo(name="manifest.json")
                m.size = len(manifest)
                tf.addfile(m, BytesIO(manifest))

                c = tarfile.TarInfo(name="config.yaml")
                c.size = len(b"model: test\n")
                tf.addfile(c, BytesIO(b"model: test\n"))

                e = tarfile.TarInfo(name=".env")
                e.size = len(b"SECRET=hacked\n")
                tf.addfile(e, BytesIO(b"SECRET=hacked\n"))

            result = _run_import_cli(str(bundle_path), "safe", str(hermes))

        assert result.returncode == 0, f"import failed: {result.stderr}"
        assert not (hermes / ".env").exists(), ".env must NOT be extracted in safe preset"
        assert (hermes / "config.yaml").exists()

    def test_import_full_preset_includes_dotenv(self, tmp_path):
        """full preset import: .env IS extracted when present in bundle."""
        hermes = tmp_path / ".hermes"

        with tempfile.TemporaryDirectory() as out_dir:
            bundle_path = Path(out_dir) / "full.tar.gz"
            with tarfile.open(bundle_path, "w:gz") as tf:
                manifest = json.dumps({
                    "version": "1.0", "preset": "full", "source_os": "linux",
                    "source_home": "/home/alice", "includes_secrets": True,
                    "bundle_created_at": "2026-01-01T00:00:00Z",
                }).encode("utf-8")
                m = tarfile.TarInfo(name="manifest.json")
                m.size = len(manifest)
                tf.addfile(m, BytesIO(manifest))

                c = tarfile.TarInfo(name="config.yaml")
                c.size = len(b"model: test\n")
                tf.addfile(c, BytesIO(b"model: test\n"))

                e_payload = b"OPENAI_KEY=sk-secret123\n"
                e = tarfile.TarInfo(name=".env")
                e.size = len(e_payload)
                tf.addfile(e, BytesIO(e_payload))

            result = _run_import_cli(str(bundle_path), "full", str(hermes))

        assert result.returncode == 0
        assert (hermes / ".env").exists()
        assert "OPENAI_KEY=sk-secret123" in (hermes / ".env").read_text(encoding="utf-8")

    def test_import_remaps_home_in_config_yaml(self, tmp_path):
        """Cross-machine: /home/alice in config.yaml → new home after import."""
        from hermes_cli.migrate import export_bundle

        hermes = tmp_path / ".hermes"
        (hermes / "config.yaml").write_text(
            "model: test\n"
            "terminal:\n"
            "  cwd: /home/alice/projects\n",
            encoding="utf-8",
        )

        with tempfile.TemporaryDirectory() as out_dir:
            out = Path(out_dir) / "bundle.tar.gz"
            export_bundle(output_path=str(out), preset="safe")

            # Patch manifest to simulate old machine home
            patched = Path(out_dir) / "patched.tar.gz"
            with tarfile.open(out, "r:gz") as src, tarfile.open(patched, "w:gz") as dst:
                for member in src.getmembers():
                    if member.name == "manifest.json":
                        new_m = json.dumps({
                            "version": "1.0", "preset": "safe", "source_os": "linux",
                            "source_home": "/home/alice",
                            "includes_secrets": False,
                            "bundle_created_at": "2026-01-01T00:00:00Z",
                        }).encode("utf-8")
                        info = tarfile.TarInfo(name="manifest.json")
                        info.size = len(new_m)
                        dst.addfile(info, BytesIO(new_m))
                    else:
                        dst.addfile(member, src.extractfile(member))

            result = _run_import_cli(str(patched), "safe", str(hermes))

        assert result.returncode == 0, f"import failed: {result.stderr}"
        config_text = (hermes / "config.yaml").read_text(encoding="utf-8")
        assert "/home/alice" not in config_text, "Old home path must be remapped"
        assert "cwd:" in config_text

    def test_import_remaps_home_in_memories(self, tmp_path):
        """Cross-machine: /home/alice in memories/ files → remapped after import."""
        from hermes_cli.migrate import export_bundle

        hermes = tmp_path / ".hermes"
        (hermes / "config.yaml").write_text("model: test\n", encoding="utf-8")
        memories = hermes / "memories"
        memories.mkdir()
        (memories / "ref.md").write_text(
            "Docs at /home/alice/docs and /home/alice/code\n", encoding="utf-8"
        )

        with tempfile.TemporaryDirectory() as out_dir:
            out = Path(out_dir) / "bundle.tar.gz"
            export_bundle(output_path=str(out), preset="safe")

            patched = Path(out_dir) / "patched.tar.gz"
            with tarfile.open(out, "r:gz") as src, tarfile.open(patched, "w:gz") as dst:
                for member in src.getmembers():
                    if member.name == "manifest.json":
                        new_m = json.dumps({
                            "version": "1.0", "preset": "safe", "source_os": "linux",
                            "source_home": "/home/alice",
                            "includes_secrets": False,
                            "bundle_created_at": "2026-01-01T00:00:00Z",
                        }).encode("utf-8")
                        info = tarfile.TarInfo(name="manifest.json")
                        info.size = len(new_m)
                        dst.addfile(info, BytesIO(new_m))
                    else:
                        dst.addfile(member, src.extractfile(member))

            result = _run_import_cli(str(patched), "safe", str(hermes))

        assert result.returncode == 0
        ref_content = (hermes / "memories" / "ref.md").read_text(encoding="utf-8")
        assert "/home/alice" not in ref_content

    def test_import_remaps_home_in_skills(self, tmp_path):
        """Cross-machine: /home/alice in skills/SKILL.md files → remapped after import."""
        from hermes_cli.migrate import export_bundle

        hermes = tmp_path / ".hermes"
        (hermes / "config.yaml").write_text("model: test\n", encoding="utf-8")
        skills = hermes / "skills"
        skills.mkdir()
        skill_dir = skills / "my-skill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text(
            "Working in /home/alice/projects\n---\nname: my-skill\n", encoding="utf-8"
        )

        with tempfile.TemporaryDirectory() as out_dir:
            out = Path(out_dir) / "bundle.tar.gz"
            export_bundle(output_path=str(out), preset="safe")

            patched = Path(out_dir) / "patched.tar.gz"
            with tarfile.open(out, "r:gz") as src, tarfile.open(patched, "w:gz") as dst:
                for member in src.getmembers():
                    if member.name == "manifest.json":
                        new_m = json.dumps({
                            "version": "1.0", "preset": "safe", "source_os": "linux",
                            "source_home": "/home/alice",
                            "includes_secrets": False,
                            "bundle_created_at": "2026-01-01T00:00:00Z",
                        }).encode("utf-8")
                        info = tarfile.TarInfo(name="manifest.json")
                        info.size = len(new_m)
                        dst.addfile(info, BytesIO(new_m))
                    else:
                        dst.addfile(member, src.extractfile(member))

            result = _run_import_cli(str(patched), "safe", str(hermes))

        assert result.returncode == 0
        skill_content = (hermes / "skills" / "my-skill" / "SKILL.md").read_text(encoding="utf-8")
        assert "/home/alice" not in skill_content

    def test_import_remaps_nested_dotenv(self, tmp_path):
        """Nested .env files inside memories/ or skills/ are also skipped in safe mode."""
        from hermes_cli.migrate import export_bundle

        hermes = tmp_path / ".hermes"
        (hermes / "config.yaml").write_text("model: test\n", encoding="utf-8")
        memories = hermes / "memories"
        memories.mkdir()
        # A .env file accidentally placed inside memories/
        (memories / "old.env").write_text("KEY=secret\n", encoding="utf-8")

        with tempfile.TemporaryDirectory() as out_dir:
            out = Path(out_dir) / "bundle.tar.gz"
            export_bundle(output_path=str(out), preset="safe")

            # Manually add a nested .env to simulate edge case
            patched = Path(out_dir) / "patched.tar.gz"
            with tarfile.open(out, "r:gz") as src, tarfile.open(patched, "w:gz") as dst:
                for member in src.getmembers():
                    dst.addfile(member, src.extractfile(member))
                # Sneak in a nested .env
                nested_env = tarfile.TarInfo(name="memories/.env")
                nested_env.size = len(b"NESTED_SECRET=abc\n")
                dst.addfile(nested_env, BytesIO(b"NESTED_SECRET=abc\n"))

            result = _run_import_cli(str(patched), "safe", str(hermes))

        assert result.returncode == 0
        # The nested .env inside memories/ should have been filtered
        nested_env_path = hermes / "memories" / ".env"
        # After remap filtering, this file should NOT exist
        assert not nested_env_path.exists(), ".env inside subdirs must not be extracted in safe mode"


class TestVerifyBundle:
    """Bundle verification."""

    def test_verify_rejects_missing_manifest(self, tmp_path):
        """verify_bundle returns False when manifest is absent."""
        from hermes_cli.migrate import verify_bundle

        hermes = tmp_path / ".hermes"

        with tempfile.TemporaryDirectory() as out_dir:
            bad = Path(out_dir) / "no-manifest.tar.gz"
            with tarfile.open(bad, "w:gz") as tf:
                c = tarfile.TarInfo(name="config.yaml")
                c.size = 5
                tf.addfile(c, BytesIO(b"test\n"))
            result = verify_bundle(input_path=str(bad))
            assert result is False

    def test_verify_rejects_invalid_yaml(self, tmp_path):
        """verify_bundle returns False when config.yaml is malformed."""
        from hermes_cli.migrate import verify_bundle

        hermes = tmp_path / ".hermes"

        with tempfile.TemporaryDirectory() as out_dir:
            bad = Path(out_dir) / "bad-yaml.tar.gz"
            with tarfile.open(bad, "w:gz") as tf:
                manifest = json.dumps({
                    "version": "1.0", "source_os": "linux", "preset": "safe",
                    "bundle_created_at": "2026-01-01T00:00:00Z",
                }).encode("utf-8")
                m = tarfile.TarInfo(name="manifest.json")
                m.size = len(manifest)
                tf.addfile(m, BytesIO(manifest))
                bad_cfg = b"model: test\n  badly: [indented\n"
                c = tarfile.TarInfo(name="config.yaml")
                c.size = len(bad_cfg)
                tf.addfile(c, BytesIO(bad_cfg))
            result = verify_bundle(input_path=str(bad))
            assert result is False

    def test_verify_accepts_valid_bundle(self, tmp_path):
        """verify_bundle returns True for a valid bundle."""
        from hermes_cli.migrate import export_bundle, verify_bundle

        hermes = tmp_path / ".hermes"
        (hermes / "config.yaml").write_text("model: test\n", encoding="utf-8")
        skills = hermes / "skills"
        skills.mkdir()
        skill_dir = skills / "test-skill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text("---\nname: test\n---\n", encoding="utf-8")

        with tempfile.TemporaryDirectory() as out_dir:
            out = export_bundle(output_path=str(Path(out_dir) / "bundle.tar.gz"), preset="safe")
            result = verify_bundle(input_path=str(out))
            assert result is True

    def test_verify_current_installation(self, tmp_path):
        """verify_bundle(None) verifies current installation."""
        from hermes_cli.migrate import verify_bundle

        hermes = tmp_path / ".hermes"
        (hermes / "config.yaml").write_text(
            "model: openrouter/auto\nproviders:\n  openrouter:\n    api_key: test-key\n",
            encoding="utf-8",
        )
        skills = hermes / "skills"
        skills.mkdir()
        skill_dir = skills / "test-skill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text("---\nname: test\n---\n", encoding="utf-8")

        result = verify_bundle(input_path=None)
        assert result is True


class TestDoctor:
    """Environment health checks."""

    def test_doctor_returns_true_healthy(self, tmp_path):
        """run_doctor returns True on a healthy installation."""
        from hermes_cli.migrate import run_doctor

        hermes = tmp_path / ".hermes"
        (hermes / "config.yaml").write_text(
            "model: openrouter/auto\nproviders:\n  openrouter:\n    api_key: real-key-here\n",
            encoding="utf-8",
        )

        result = run_doctor()
        assert result is True


# ============================================================================
# Task 3 completion: missing single-direction tests
# ============================================================================


class TestImportExistingFiles:
    """Import handles existing files by backing them up before overwriting."""

    def test_import_overwrites_existing_file(self, tmp_path):
        """Existing file is overwritten with bundle content (old version backed up)."""
        from hermes_cli.migrate import export_bundle

        hermes = tmp_path / ".hermes"
        (hermes / "config.yaml").write_text("model: test\n", encoding="utf-8")
        (hermes / "SOUL.md").write_text("# Old Soul\n", encoding="utf-8")

        with tempfile.TemporaryDirectory() as out_dir:
            out = Path(out_dir) / "bundle.tar.gz"
            export_bundle(output_path=str(out), preset="safe")

            # Modify file before re-importing
            (hermes / "config.yaml").write_text("modified\n", encoding="utf-8")
            result = _run_import_cli(str(out), "safe", str(hermes))

        assert result.returncode == 0, f"import failed: {result.stderr}"
        assert (hermes / "config.yaml").read_text(encoding="utf-8") == "model: test\n"
        backup_dirs = list(hermes.glob(".hermes.bak*"))
        assert len(backup_dirs) == 1

    def test_import_preserves_unbundled_files(self, tmp_path):
        """Files that exist on disk but are NOT in the bundle are left alone."""
        from hermes_cli.migrate import export_bundle

        hermes = tmp_path / ".hermes"
        (hermes / "config.yaml").write_text("model: test\n", encoding="utf-8")
        (hermes / "local-only.txt").write_text("not in bundle\n", encoding="utf-8")

        with tempfile.TemporaryDirectory() as out_dir:
            out = Path(out_dir) / "bundle.tar.gz"
            export_bundle(output_path=str(out), preset="safe")
            result = _run_import_cli(str(out), "safe", str(hermes))

        assert result.returncode == 0
        assert (hermes / "local-only.txt").exists()
        assert (hermes / "local-only.txt").read_text(encoding="utf-8") == "not in bundle\n"


class TestImportProfilesAndSessions:
    """Profiles and sessions directory import."""

    def test_import_profiles_subdirectory_contents(self, tmp_path):
        """Profiles subdirectory files are extracted correctly."""
        from hermes_cli.migrate import export_bundle

        hermes = tmp_path / ".hermes"
        (hermes / "config.yaml").write_text("model: test\n", encoding="utf-8")
        profiles = hermes / "profiles"
        profiles.mkdir()
        coder = profiles / "coder"
        coder.mkdir()
        (coder / "config.yaml").write_text("model: coder-model\n", encoding="utf-8")
        (coder / "SOUL.md").write_text("# Coder Soul\n", encoding="utf-8")

        with tempfile.TemporaryDirectory() as out_dir:
            out = Path(out_dir) / "bundle.tar.gz"
            export_bundle(output_path=str(out), preset="safe")
            result = _run_import_cli(str(out), "safe", str(hermes))

        assert result.returncode == 0, f"import failed: {result.stderr}"
        assert (hermes / "profiles" / "coder" / "config.yaml").read_text(encoding="utf-8") == "model: coder-model\n"
        assert (hermes / "profiles" / "coder" / "SOUL.md").exists()

    def test_import_sessions_files(self, tmp_path):
        """Session files are imported correctly."""
        from hermes_cli.migrate import export_bundle

        hermes = tmp_path / ".hermes"
        (hermes / "config.yaml").write_text("model: test\n", encoding="utf-8")
        sessions = hermes / "sessions"
        sessions.mkdir()
        (sessions / "session-abc.json").write_text('{"id": "abc", "messages": []}', encoding="utf-8")
        (sessions / "session-xyz.json").write_text('{"id": "xyz", "messages": [{"role": "user"}]}', encoding="utf-8")

        with tempfile.TemporaryDirectory() as out_dir:
            out = Path(out_dir) / "bundle.tar.gz"
            export_bundle(output_path=str(out), preset="safe")
            result = _run_import_cli(str(out), "safe", str(hermes))

        assert result.returncode == 0, f"import failed: {result.stderr}"
        content = (hermes / "sessions" / "session-abc.json").read_text(encoding="utf-8")
        assert '"id": "abc"' in content


class TestPathRemappingCoverage:
    """Additional path remapping for docker_volumes and external_dirs."""

    def _inject_source_home(self, bundle_path: Path, out_dir: Path, source_home: str) -> Path:
        """Patch a bundle's manifest to set a specific source_home."""
        patched = out_dir / "patched.tar.gz"
        with tarfile.open(bundle_path, "r:gz") as src, tarfile.open(patched, "w:gz") as dst:
            for member in src.getmembers():
                if member.name == "manifest.json":
                    new_m = json.dumps({
                        "version": "1.0", "preset": "safe", "source_os": "linux",
                        "source_home": source_home,
                        "includes_secrets": False,
                        "bundle_created_at": "2026-01-01T00:00:00Z",
                    }).encode("utf-8")
                    info = tarfile.TarInfo(name="manifest.json")
                    info.size = len(new_m)
                    dst.addfile(info, BytesIO(new_m))
                else:
                    dst.addfile(member, src.extractfile(member))
        return patched

    def test_import_remaps_docker_volumes(self, tmp_path):
        """docker_volumes paths in config.yaml are remapped."""
        from hermes_cli.migrate import export_bundle

        hermes = tmp_path / ".hermes"
        (hermes / "config.yaml").write_text(
            "model: test\n"
            "terminal:\n"
            "  docker_volumes:\n"
            "    - /home/alice/data:/container/data\n"
            "    - /home/alice/cache:/container/cache\n",
            encoding="utf-8",
        )

        with tempfile.TemporaryDirectory() as out_dir:
            out = Path(out_dir) / "bundle.tar.gz"
            export_bundle(output_path=str(out), preset="safe")
            patched = self._inject_source_home(out, Path(out_dir), "/home/alice")
            result = _run_import_cli(str(patched), "safe", str(hermes))

        assert result.returncode == 0, f"import failed: {result.stderr}"
        config = (hermes / "config.yaml").read_text(encoding="utf-8")
        assert "/home/alice" not in config
        assert "/container/data" in config

    def test_import_remaps_external_dirs(self, tmp_path):
        """external_dirs paths in config.yaml are remapped."""
        from hermes_cli.migrate import export_bundle

        hermes = tmp_path / ".hermes"
        (hermes / "config.yaml").write_text(
            "model: test\nexternal_dirs:\n  - /home/alice/projects\n  - /home/alice/dotfiles\n",
            encoding="utf-8",
        )

        with tempfile.TemporaryDirectory() as out_dir:
            out = Path(out_dir) / "bundle.tar.gz"
            export_bundle(output_path=str(out), preset="safe")
            patched = self._inject_source_home(out, Path(out_dir), "/home/alice")
            result = _run_import_cli(str(patched), "safe", str(hermes))

        assert result.returncode == 0, f"import failed: {result.stderr}"
        config = (hermes / "config.yaml").read_text(encoding="utf-8")
        assert "/home/alice" not in config

    def test_import_remaps_deeply_nested_memories(self, tmp_path):
        """Deeply nested memories files get path remapped too."""
        from hermes_cli.migrate import export_bundle

        hermes = tmp_path / ".hermes"
        (hermes / "config.yaml").write_text("model: test\n", encoding="utf-8")
        memories = hermes / "memories"
        memories.mkdir()
        deep = memories / "projects" / "myapp" / "docs"
        deep.mkdir(parents=True)
        (deep / "ref.md").write_text("See /home/alice/projects/myapp/README\n", encoding="utf-8")

        with tempfile.TemporaryDirectory() as out_dir:
            out = Path(out_dir) / "bundle.tar.gz"
            export_bundle(output_path=str(out), preset="safe")
            patched = self._inject_source_home(out, Path(out_dir), "/home/alice")
            result = _run_import_cli(str(patched), "safe", str(hermes))

        assert result.returncode == 0
        ref = (hermes / "memories" / "projects" / "myapp" / "docs" / "ref.md").read_text(encoding="utf-8")
        assert "/home/alice" not in ref


class TestSafePresetSecretExclusion:
    """Additional safe preset exclusion edge cases."""

    def test_export_safe_excludes_nested_dotenv_in_memories(self, tmp_path):
        """safe preset: .env nested inside memories/ is not in the bundle."""
        from hermes_cli.migrate import export_bundle

        hermes = tmp_path / ".hermes"
        (hermes / "config.yaml").write_text("model: test\n", encoding="utf-8")
        memories = hermes / "memories"
        memories.mkdir()
        (memories / "notes.md").write_text("# Notes\n", encoding="utf-8")
        (memories / ".env").write_text("SECRET=inside_memories\n", encoding="utf-8")

        with tempfile.TemporaryDirectory() as out_dir:
            out = Path(out_dir) / "bundle.tar.gz"
            export_bundle(output_path=str(out), preset="safe")
            with tarfile.open(out, "r:gz") as tf:
                names = tf.getnames()

        assert ".env" not in names
        assert "memories/.env" not in names
        assert "memories/notes.md" in names

    def test_export_safe_excludes_nested_dotenv_in_skills(self, tmp_path):
        """safe preset: .env nested inside skills/ is not in the bundle."""
        from hermes_cli.migrate import export_bundle

        hermes = tmp_path / ".hermes"
        (hermes / "config.yaml").write_text("model: test\n", encoding="utf-8")
        skills = hermes / "skills"
        skills.mkdir()
        skill_dir = skills / "my-skill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text("---\nname: my-skill\n---\n", encoding="utf-8")
        (skill_dir / ".env").write_text("INTERNAL=key\n", encoding="utf-8")

        with tempfile.TemporaryDirectory() as out_dir:
            out = Path(out_dir) / "bundle.tar.gz"
            export_bundle(output_path=str(out), preset="safe")
            with tarfile.open(out, "r:gz") as tf:
                names = tf.getnames()

        assert not any(".env" in n for n in names)


# ============================================================================
# Task 4: End-to-end round-trip tests — export → import → verify
# ============================================================================


class TestRoundTrip:
    """Full export → import round-trip tests."""

    def test_roundtrip_config_and_memories(self, tmp_path):
        """Export then import preserves config.yaml and memories content exactly."""
        from hermes_cli.migrate import export_bundle

        hermes = tmp_path / ".hermes"
        (hermes / "config.yaml").write_text(
            "model: openrouter/奥特曼\nproviders:\n  openrouter:\n    api_key: sk-test\n",
            encoding="utf-8",
        )
        (hermes / "SOUL.md").write_text("# 自定义灵魂\n", encoding="utf-8")
        memories = hermes / "memories"
        memories.mkdir()
        (memories / "idea.md").write_text("# 新想法\n", encoding="utf-8")

        with tempfile.TemporaryDirectory() as out_dir:
            out = Path(out_dir) / "bundle.tar.gz"
            export_bundle(output_path=str(out), preset="safe")

            import shutil
            if (hermes / "memories").exists():
                shutil.rmtree(hermes / "memories")
            if (hermes / "SOUL.md").exists():
                (hermes / "SOUL.md").unlink()

            result = _run_import_cli(str(out), "safe", str(hermes))
            assert result.returncode == 0, f"import failed: {result.stderr}"

        cfg = (hermes / "config.yaml").read_text(encoding="utf-8")
        assert "openrouter/奥特曼" in cfg
        assert (hermes / "SOUL.md").exists()
        memories_content = (hermes / "memories" / "idea.md").read_text(encoding="utf-8")
        assert "# 新想法" in memories_content

    def test_roundtrip_skills_preserved(self, tmp_path):
        """Skills with nested files survive a full export → import cycle."""
        from hermes_cli.migrate import export_bundle

        hermes = tmp_path / ".hermes"
        (hermes / "config.yaml").write_text("model: test\n", encoding="utf-8")
        skills = hermes / "skills"
        skills.mkdir()
        skill_dir = skills / "devops-skill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text(
            "---\nname: devops-skill\ndescription: DevOps workflows\n---\n# DevOps\n",
            encoding="utf-8",
        )
        refs = skill_dir / "references"
        refs.mkdir()
        (refs / "docker.md").write_text("# Docker Reference\n", encoding="utf-8")
        (refs / "k8s.md").write_text("# Kubernetes Reference\n", encoding="utf-8")

        with tempfile.TemporaryDirectory() as out_dir:
            out = Path(out_dir) / "bundle.tar.gz"
            export_bundle(output_path=str(out), preset="safe")
            import shutil
            shutil.rmtree(skills)
            result = _run_import_cli(str(out), "safe", str(hermes))
            assert result.returncode == 0, f"import failed: {result.stderr}"

        assert (hermes / "skills" / "devops-skill" / "references" / "docker.md").exists()
        content = (hermes / "skills" / "devops-skill" / "references" / "docker.md").read_text(encoding="utf-8")
        assert "Docker Reference" in content

    def test_roundtrip_profiles(self, tmp_path):
        """Multiple profiles survive export → import."""
        from hermes_cli.migrate import export_bundle

        hermes = tmp_path / ".hermes"
        (hermes / "config.yaml").write_text("model: test\n", encoding="utf-8")
        profiles = hermes / "profiles"
        profiles.mkdir()
        for name in ["coder", "researcher", "admin"]:
            profile_dir = profiles / name
            profile_dir.mkdir()
            (profile_dir / "config.yaml").write_text(f"model: {name}-model\n", encoding="utf-8")
            (profile_dir / "SOUL.md").write_text(f"# {name} soul\n", encoding="utf-8")

        with tempfile.TemporaryDirectory() as out_dir:
            out = Path(out_dir) / "bundle.tar.gz"
            export_bundle(output_path=str(out), preset="safe")
            import shutil
            shutil.rmtree(profiles)
            result = _run_import_cli(str(out), "safe", str(hermes))
            assert result.returncode == 0, f"import failed: {result.stderr}"

        for name in ["coder", "researcher", "admin"]:
            cfg = (hermes / "profiles" / name / "config.yaml").read_text(encoding="utf-8")
            assert f"{name}-model" in cfg

    def test_roundtrip_cross_machine_path_remap(self, tmp_path):
        """Paths from a different machine are correctly remapped on import."""
        from hermes_cli.migrate import export_bundle

        hermes = tmp_path / ".hermes"
        (hermes / "config.yaml").write_text(
            "model: test\n"
            "terminal:\n"
            "  cwd: /home/alice/projects\n"
            "external_dirs:\n"
            "  - /home/alice/dotfiles\n",
            encoding="utf-8",
        )
        memories = hermes / "memories"
        memories.mkdir()
        (memories / "note.md").write_text("Projects at /home/alice/projects\n", encoding="utf-8")

        with tempfile.TemporaryDirectory() as out_dir:
            out = Path(out_dir) / "bundle.tar.gz"
            export_bundle(output_path=str(out), preset="safe")

            # Inject /home/alice as source_home
            patched = Path(out_dir) / "patched.tar.gz"
            with tarfile.open(out, "r:gz") as src, tarfile.open(patched, "w:gz") as dst:
                for member in src.getmembers():
                    if member.name == "manifest.json":
                        new_m = json.dumps({
                            "version": "1.0", "preset": "safe", "source_os": "linux",
                            "source_home": "/home/alice",
                            "includes_secrets": False,
                            "bundle_created_at": "2026-01-01T00:00:00Z",
                        }).encode("utf-8")
                        info = tarfile.TarInfo(name="manifest.json")
                        info.size = len(new_m)
                        dst.addfile(info, BytesIO(new_m))
                    else:
                        dst.addfile(member, src.extractfile(member))

            result = _run_import_cli(str(patched), "safe", str(hermes))
            assert result.returncode == 0, f"import failed: {result.stderr}"

        config = (hermes / "config.yaml").read_text(encoding="utf-8")
        assert "/home/alice" not in config
        memories_content = (hermes / "memories" / "note.md").read_text(encoding="utf-8")
        assert "/home/alice" not in memories_content

    def test_roundtrip_safe_preset_never_leaks_secrets(self, tmp_path):
        """Even if .env exists on source, safe preset export never includes it."""
        from hermes_cli.migrate import export_bundle

        hermes = tmp_path / ".hermes"
        (hermes / "config.yaml").write_text("model: test\n", encoding="utf-8")
        (hermes / ".env").write_text("OPENAI_KEY=sk-super-secret-key\n", encoding="utf-8")
        memories = hermes / "memories"
        memories.mkdir()
        (memories / "note.md").write_text("# Note\n", encoding="utf-8")

        with tempfile.TemporaryDirectory() as out_dir:
            out = Path(out_dir) / "bundle.tar.gz"
            export_bundle(output_path=str(out), preset="safe")

            with tarfile.open(out, "r:gz") as tf:
                names = tf.getnames()
            assert ".env" not in names, "safe preset must not include .env in bundle"

            if (hermes / ".env").exists():
                (hermes / ".env").unlink()
            import shutil
            if (hermes / "memories").exists():
                shutil.rmtree(hermes / "memories")

            result = _run_import_cli(str(out), "safe", str(hermes))
            assert result.returncode == 0

        assert not (hermes / ".env").exists()
        assert (hermes / "memories" / "note.md").exists()
