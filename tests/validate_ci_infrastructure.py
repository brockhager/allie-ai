#!/usr/bin/env python3
"""
CI/CD Test Infrastructure Validation Script

This script validates that the entire test infrastructure runs reliably
in a CI/CD environment by running all test suites and checking for consistency.
"""

import subprocess
import sys
import time
import json
from pathlib import Path


def run_command(cmd, description, timeout=120):
    """Run a command and return success/failure with output"""
    print(f"\n{'='*60}")
    print(f"🔍 {description}")
    print(f"Command: {cmd}")
    print('='*60)
    
    start_time = time.time()
    try:
        result = subprocess.run(
            cmd, 
            shell=True, 
            capture_output=True, 
            text=True, 
            timeout=timeout,
            cwd=Path(__file__).parent.parent
        )
        
        duration = time.time() - start_time
        
        if result.returncode == 0:
            print(f"✅ SUCCESS ({duration:.1f}s)")
            if result.stdout:
                print("STDOUT:", result.stdout[-500:])  # Last 500 chars
            return True, result.stdout, result.stderr
        else:
            print(f"❌ FAILED ({duration:.1f}s) - Return code: {result.returncode}")
            if result.stdout:
                print("STDOUT:", result.stdout[-1000:])  # Last 1000 chars on failure
            if result.stderr:
                print("STDERR:", result.stderr[-1000:])
            return False, result.stdout, result.stderr
            
    except subprocess.TimeoutExpired:
        print(f"⏰ TIMEOUT after {timeout}s")
        return False, "", f"Command timed out after {timeout}s"
    except Exception as e:
        print(f"💥 ERROR: {e}")
        return False, "", str(e)


def validate_test_infrastructure():
    """Comprehensive test infrastructure validation"""
    
    print("🚀 Starting CI/CD Test Infrastructure Validation")
    print(f"Working directory: {Path(__file__).parent.parent}")
    
    results = {}
    total_tests = 0
    
    # Test configurations
    test_suites = [
        {
            "name": "Server API Tests",
            "command": "python -m pytest tests/test_server_api.py -v --tb=short",
            "timeout": 60,
            "critical": True
        },
        {
            "name": "Hybrid Memory Tests", 
            "command": "python -m pytest tests/test_hybrid_memory.py -v --tb=short",
            "timeout": 60,
            "critical": True
        },
        {
            "name": "Integration Tests",
            "command": "python -m pytest tests/test_integration.py -v --tb=short", 
            "timeout": 90,
            "critical": True
        },
        {
            "name": "New API Endpoints Tests",
            "command": "python -m pytest tests/test_api_endpoints_integration.py -v --tb=short",
            "timeout": 60,
            "critical": True
        },
        {
            "name": "All Tests (Quick Run)",
            "command": "python -m pytest tests/ -x --tb=short -q",
            "timeout": 180,
            "critical": False
        },
        {
            "name": "Import Validation",
            "command": "python -c \"from backend.server import app; from backend.automatic_learner import AutomaticLearner; print('All critical imports successful')\"",
            "timeout": 30,
            "critical": True
        }
    ]
    
    # Run each test suite
    for i, suite in enumerate(test_suites, 1):
        success, stdout, stderr = run_command(
            suite["command"],
            f"[{i}/{len(test_suites)}] {suite['name']}",
            suite["timeout"]
        )
        
        results[suite["name"]] = {
            "success": success,
            "stdout": stdout,
            "stderr": stderr,
            "critical": suite["critical"]
        }
        
        # Extract test count from pytest output
        if "passed" in stdout and "test" in stdout:
            try:
                # Look for patterns like "12 passed" or "5 failed, 7 passed"
                import re
                matches = re.findall(r'(\d+) passed', stdout)
                if matches:
                    total_tests += int(matches[-1])
            except:
                pass
        
        # Stop if critical test fails
        if not success and suite["critical"]:
            print(f"\n❌ CRITICAL TEST FAILURE: {suite['name']}")
            print("Stopping validation due to critical failure.")
            break
    
    # Summary report
    print("\n" + "="*80)
    print("📊 CI/CD VALIDATION SUMMARY")
    print("="*80)
    
    passed_suites = sum(1 for r in results.values() if r["success"])
    total_suites = len(results)
    critical_failures = sum(1 for r in results.values() if not r["success"] and r["critical"])
    
    print(f"Test Suites: {passed_suites}/{total_suites} passed")
    print(f"Total Tests Run: ~{total_tests}")
    print(f"Critical Failures: {critical_failures}")
    
    for name, result in results.items():
        status = "✅ PASS" if result["success"] else "❌ FAIL"
        critical = " (CRITICAL)" if result["critical"] else ""
        print(f"  {status} {name}{critical}")
    
    # Overall assessment
    print("\n" + "="*80)
    if critical_failures == 0 and passed_suites == total_suites:
        print("🎉 CI/CD VALIDATION: EXCELLENT")
        print("✅ All test suites passed")
        print("✅ No critical failures")
        print("✅ Infrastructure is production-ready")
        return True
    elif critical_failures == 0:
        print("✅ CI/CD VALIDATION: GOOD")
        print("✅ All critical tests passed")
        print(f"⚠️  {total_suites - passed_suites} non-critical tests failed")
        print("✅ Infrastructure is suitable for deployment")
        return True
    else:
        print("❌ CI/CD VALIDATION: FAILED")
        print(f"❌ {critical_failures} critical test(s) failed")
        print("❌ Infrastructure needs fixes before deployment")
        return False


def generate_ci_config():
    """Generate a sample CI/CD configuration"""
    
    github_actions_config = {
        "name": "Allie AI Test Suite",
        "on": {
            "push": {"branches": ["main", "develop"]},
            "pull_request": {"branches": ["main"]}
        },
        "jobs": {
            "test": {
                "runs-on": "ubuntu-latest",
                "strategy": {
                    "matrix": {
                        "python-version": ["3.10", "3.11", "3.12"]
                    }
                },
                "steps": [
                    {"uses": "actions/checkout@v4"},
                    {
                        "name": "Set up Python ${{ matrix.python-version }}",
                        "uses": "actions/setup-python@v4",
                        "with": {"python-version": "${{ matrix.python-version }}"}
                    },
                    {
                        "name": "Install dependencies",
                        "run": "pip install -r requirements-test.txt"
                    },
                    {
                        "name": "Run test suite",
                        "run": "python tests/validate_ci_infrastructure.py"
                    },
                    {
                        "name": "Generate coverage report",
                        "run": "python -m pytest --cov=backend --cov-report=xml tests/"
                    },
                    {
                        "name": "Upload coverage reports",
                        "uses": "codecov/codecov-action@v3"
                    }
                ]
            }
        }
    }
    
    # Save CI configuration
    ci_dir = Path(__file__).parent.parent / ".github" / "workflows"
    ci_dir.mkdir(parents=True, exist_ok=True)
    
    ci_file = ci_dir / "test.yml"
    with open(ci_file, 'w') as f:
        import yaml
        yaml.dump(github_actions_config, f, default_flow_style=False, sort_keys=False)
    
    print(f"\n📝 Generated CI/CD configuration: {ci_file}")
    return ci_file


if __name__ == "__main__":
    try:
        # Run validation
        success = validate_test_infrastructure()
        
        # Generate CI config if validation passes
        if success:
            try:
                generate_ci_config()
            except Exception as e:
                print(f"⚠️  Could not generate CI config: {e}")
        
        # Exit with appropriate code
        sys.exit(0 if success else 1)
        
    except KeyboardInterrupt:
        print("\n⏹️  Validation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Validation failed with error: {e}")
        sys.exit(1)