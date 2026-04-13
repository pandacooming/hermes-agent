"""
Hermes migration export command.

Creates a migration bundle from current Hermes installation.
"""

import json
import os
import tarfile
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Optional

from hermes_cli.colors import Colors, color
from hermes_cli.migrate_core import (
    HERMES_HOME,
    _collect_migration_items,
    _should_skip_dir,
    _should_skip_file,
    create_manifest,
    detect_platform,
    _log_warning,
)


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
                            # Secret files (nested at any depth) are excluded by safe preset
                            if preset != "full" and fname in {".env", "auth.json"}:
                                continue
                            if _should_skip_file(fname, source_platform["os"]):
                                continue
                            full_path = Path(parent) / fname
                            arcname = full_path.relative_to(HERMES_HOME).as_posix()
                            tf.add(str(full_path), arcname=arcname)
                            migrated_count += 1
                else:
                    if _should_skip_file(rel_path, source_platform["os"]):
                        continue
                    tf.add(str(src), arcname=rel_path)
                    migrated_count += 1
            except (OSError, IOError) as e:
                _log_warning(f"Could not add {rel_path}: {e}")

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
