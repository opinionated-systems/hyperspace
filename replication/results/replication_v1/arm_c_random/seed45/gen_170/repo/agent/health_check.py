"""
Health check utilities for verifying agent setup and configuration.

Provides functions to validate that the agent environment is properly
configured and all dependencies are available.
"""

from __future__ import annotations

import importlib
import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class HealthCheckResult:
    """Result of a health check."""
    
    name: str
    passed: bool
    message: str
    details: dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "name": self.name,
            "passed": self.passed,
            "message": self.message,
            "details": self.details,
        }


class HealthChecker:
    """Run health checks on the agent environment."""
    
    def __init__(self) -> None:
        self.results: list[HealthCheckResult] = []
    
    def check_all(self) -> list[HealthCheckResult]:
        """Run all health checks."""
        self.results = []
        
        self._check_python_version()
        self._check_required_packages()
        self._check_environment_variables()
        self._check_tools()
        self._check_file_permissions()
        
        return self.results
    
    def _check_python_version(self) -> None:
        """Check Python version compatibility."""
        version = sys.version_info
        min_version = (3, 9)
        
        passed = version >= min_version
        message = f"Python {version.major}.{version.minor}.{version.micro}"
        if not passed:
            message += f" (requires >= {min_version[0]}.{min_version[1]})"
        
        self.results.append(HealthCheckResult(
            name="Python Version",
            passed=passed,
            message=message,
            details={"version": f"{version.major}.{version.minor}.{version.micro}"},
        ))
    
    def _check_required_packages(self) -> None:
        """Check that required packages are installed."""
        required = [
            "markspace.llm",
            "dotenv",
        ]
        
        missing = []
        for package in required:
            try:
                importlib.import_module(package)
            except ImportError:
                missing.append(package)
        
        passed = len(missing) == 0
        message = "All required packages installed" if passed else f"Missing: {', '.join(missing)}"
        
        self.results.append(HealthCheckResult(
            name="Required Packages",
            passed=passed,
            message=message,
            details={"missing": missing},
        ))
    
    def _check_environment_variables(self) -> None:
        """Check that required environment variables are set."""
        # These are typically set via .env file
        optional_vars = [
            "LLM_MAX_TOKENS",
            "LLM_TEMPERATURE",
            "LLM_MAX_RETRIES",
            "LLM_TIMEOUT",
        ]
        
        set_vars = []
        unset_vars = []
        
        for var in optional_vars:
            if os.environ.get(var):
                set_vars.append(var)
            else:
                unset_vars.append(var)
        
        # All are optional, so this always passes
        # but we report which are using defaults
        message = f"{len(set_vars)} custom, {len(unset_vars)} using defaults"
        
        self.results.append(HealthCheckResult(
            name="Environment Variables",
            passed=True,
            message=message,
            details={"set": set_vars, "using_defaults": unset_vars},
        ))
    
    def _check_tools(self) -> None:
        """Check that tools can be loaded."""
        try:
            from agent.tools import load_tools
            tools = load_tools("all")
            tool_names = [t["info"]["name"] for t in tools]
            
            passed = len(tools) > 0
            message = f"Loaded {len(tools)} tools: {', '.join(tool_names)}"
            
            self.results.append(HealthCheckResult(
                name="Tools Loadable",
                passed=passed,
                message=message,
                details={"tools": tool_names},
            ))
        except Exception as e:
            self.results.append(HealthCheckResult(
                name="Tools Loadable",
                passed=False,
                message=f"Failed to load tools: {e}",
                details={"error": str(e)},
            ))
    
    def _check_file_permissions(self) -> None:
        """Check that we have write permissions in the working directory."""
        test_file = Path(".health_check_test")
        try:
            test_file.write_text("test")
            test_file.unlink()
            
            self.results.append(HealthCheckResult(
                name="File Permissions",
                passed=True,
                message="Write access confirmed",
                details={"cwd": str(Path.cwd())},
            ))
        except Exception as e:
            self.results.append(HealthCheckResult(
                name="File Permissions",
                passed=False,
                message=f"No write access: {e}",
                details={"cwd": str(Path.cwd()), "error": str(e)},
            ))
    
    def get_summary(self) -> dict[str, Any]:
        """Get a summary of all health checks."""
        passed = sum(1 for r in self.results if r.passed)
        total = len(self.results)
        
        return {
            "total_checks": total,
            "passed": passed,
            "failed": total - passed,
            "healthy": passed == total,
            "checks": [r.to_dict() for r in self.results],
        }
    
    def print_report(self) -> None:
        """Print a formatted health check report."""
        summary = self.get_summary()
        
        print("=" * 60)
        print("AGENT HEALTH CHECK REPORT")
        print("=" * 60)
        
        for result in self.results:
            status = "✓ PASS" if result.passed else "✗ FAIL"
            print(f"\n[{status}] {result.name}")
            print(f"  {result.message}")
            if result.details:
                for key, value in result.details.items():
                    if isinstance(value, list) and value:
                        print(f"  - {key}: {', '.join(str(v) for v in value)}")
        
        print("\n" + "=" * 60)
        if summary["healthy"]:
            print(f"All {summary['total_checks']} checks passed ✓")
        else:
            print(f"{summary['failed']}/{summary['total_checks']} checks failed")
        print("=" * 60)


def run_health_check() -> dict[str, Any]:
    """Run health checks and return results.
    
    Returns:
        Dictionary with health check summary
    """
    checker = HealthChecker()
    checker.check_all()
    return checker.get_summary()


if __name__ == "__main__":
    checker = HealthChecker()
    checker.check_all()
    checker.print_report()
