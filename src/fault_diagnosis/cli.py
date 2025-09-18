"""Command-line interface for the telecom_ops demo application."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Optional

from .workflow.orchestrator import SimpleFaultDiagnosisRunner, SimpleWorkflowSettings


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="telecom-ops", description="Telecom operations demo CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    fault_parser = subparsers.add_parser(
        "fault-diagnosis",
        help="Run the simple fault diagnosis MVP workflow",
    )
    fault_parser.add_argument(
        "--session",
        metavar="NAME",
        default=None,
        help="Optional session label for the run",
    )
    fault_parser.add_argument(
        "--no-rag",
        action="store_true",
        help="Disable RAG pipeline (for testing)",
    )
    fault_parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce verbose output",
    )

    return parser


def handle_fault_diagnosis(args: argparse.Namespace) -> int:
    project_root = Path(__file__).resolve().parent.parent.parent
    settings = SimpleWorkflowSettings(
        project_root=project_root,
        session_label=args.session,
        use_rag=not args.no_rag,
        verbose=not args.quiet,
    )
    runner = SimpleFaultDiagnosisRunner(settings)

    print("Starting Simple Fault Diagnosis MVP")
    print("=" * 40)

    # Show component status
    status = runner.get_status()
    print("\nComponent Status:")
    for component, available in status.items():
        icon = "[OK]" if available else "[FAIL]"
        print(f"  {icon} {component}: {'Available' if available else 'Not Available'}")

    print("\nRunning workflow...")
    results = runner.run()

    print("\nMVP Demo complete!")
    try:
        print(f"Results: {results}")
    except UnicodeEncodeError:
        print("Results: [Unicode encoding issue - workflow completed successfully]")
    return 0


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "fault-diagnosis":
        return handle_fault_diagnosis(args)
    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
