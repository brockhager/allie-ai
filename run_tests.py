#!/usr/bin/env python3
"""
Test runner script for Allie AI system.

This script provides convenient commands to run different types of tests
and generate coverage reports.
"""

import subprocess
import sys
import argparse
from pathlib import Path

def run_command(cmd, description):
    """Run a command and return the result."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print('='*60)

    try:
        result = subprocess.run(cmd, cwd=Path(__file__).parent)
        return result.returncode == 0
    except KeyboardInterrupt:
        print("\nTest run interrupted by user")
        return False
    except Exception as e:
        print(f"Error running command: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Allie AI Test Runner")
    parser.add_argument(
        "command",
        choices=["unit", "integration", "all", "coverage", "quick"],
        help="Test command to run"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )
    parser.add_argument(
        "--no-cov",
        action="store_true",
        help="Skip coverage reporting"
    )

    args = parser.parse_args()

    # Base pytest command
    base_cmd = [sys.executable, "-m", "pytest"]

    if args.verbose:
        base_cmd.append("-v")

    # Add coverage if not disabled
    if not args.no_cov:
        base_cmd.extend([
            "--cov=backend",
            "--cov=advanced_memory",
            "--cov-report=term-missing",
            "--cov-report=html:htmlcov"
        ])

    success = True

    if args.command == "unit":
        # Run only unit tests
        cmd = base_cmd + [
            "-m", "not integration",
            "tests/test_server_api.py",
            "tests/test_hybrid_memory.py"
        ]
        success &= run_command(cmd, "Unit Tests")

    elif args.command == "integration":
        # Run only integration tests
        cmd = base_cmd + [
            "-m", "integration",
            "tests/test_integration.py"
        ]
        success &= run_command(cmd, "Integration Tests")

    elif args.command == "all":
        # Run all tests
        cmd = base_cmd + ["tests/"]
        success &= run_command(cmd, "All Tests")

    elif args.command == "coverage":
        # Run with detailed coverage
        cmd = base_cmd + [
            "--cov-report=html:htmlcov",
            "--cov-report=xml",
            "tests/"
        ]
        success &= run_command(cmd, "Coverage Analysis")

        if success:
            print("\n" + "="*60)
            print("Coverage report generated:")
            print("  HTML: htmlcov/index.html")
            print("  XML: coverage.xml")
            print("="*60)

    elif args.command == "quick":
        # Quick test run without coverage
        cmd = [sys.executable, "-m", "pytest", "tests/", "--tb=short"]
        success &= run_command(cmd, "Quick Test Run")

    if success:
        print(f"\n✅ {args.command.title()} tests completed successfully!")
        return 0
    else:
        print(f"\n❌ {args.command.title()} tests failed!")
        return 1

if __name__ == "__main__":
    sys.exit(main())