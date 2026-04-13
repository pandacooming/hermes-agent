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

import sys
from pathlib import Path
from typing import Optional

from hermes_cli.colors import Colors, color
from hermes_cli.migrate_core import MigrationReport
from hermes_cli.migrate_export import export_bundle
from hermes_cli.migrate_import import import_bundle
from hermes_cli.migrate_verify import run_doctor, verify_bundle


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
    epilog = """
Examples:
  hermes migrate export                       Export to hermes-migration-{timestamp}.tar.gz
  hermes migrate export --preset full         Include secrets (.env, auth.json)
  hermes migrate export -o /tmp/backup.tar.gz Custom output path
  hermes migrate import -i backup.tar.gz      Import from bundle
  hermes migrate verify -i backup.tar.gz      Verify bundle
  hermes migrate doctor                       Check environment health
"""
    parser = argparse.ArgumentParser(
        "hermes migrate",
        description="Unified migration command for Hermes Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=epilog,
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
