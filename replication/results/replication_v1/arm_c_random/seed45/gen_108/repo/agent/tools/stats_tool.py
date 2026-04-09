"""
Stats tool: get system statistics and resource information.

Provides CPU, memory, disk usage, and process information.
Useful for understanding the current system state and resource constraints.
"""

from __future__ import annotations

import os
import subprocess


def tool_info() -> dict:
    return {
        "name": "stats",
        "description": (
            "Get system statistics and resource information. "
            "Provides CPU, memory, disk usage, and process info. "
            "Useful for understanding system state and constraints."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "category": {
                    "type": "string",
                    "enum": ["cpu", "memory", "disk", "processes", "all"],
                    "description": "Category of stats to retrieve (default: all)",
                }
            },
        },
    }


def _get_cpu_info() -> str:
    """Get CPU usage information."""
    try:
        # Get CPU count
        cpu_count = os.cpu_count() or "unknown"
        
        # Get load average
        load_avg = os.getloadavg() if hasattr(os, 'getloadavg') else ("N/A", "N/A", "N/A")
        
        # Try to get current CPU usage via top
        result = subprocess.run(
            ["top", "-bn1", "-p", "0"],
            capture_output=True,
            text=True,
            timeout=2
        )
        
        lines = result.stdout.split("\n")
        cpu_line = ""
        for line in lines:
            if "Cpu(s)" in line or "%Cpu" in line:
                cpu_line = line.strip()
                break
        
        return (
            f"CPU Info:\n"
            f"  CPU Count: {cpu_count}\n"
            f"  Load Average (1/5/15 min): {load_avg[0]:.2f}, {load_avg[1]:.2f}, {load_avg[2]:.2f}\n"
            f"  {cpu_line if cpu_line else 'CPU usage data unavailable'}"
        )
    except Exception as e:
        return f"CPU Info: Error retrieving - {e}"


def _get_memory_info() -> str:
    """Get memory usage information."""
    try:
        result = subprocess.run(
            ["free", "-h"],
            capture_output=True,
            text=True,
            timeout=2
        )
        
        if result.returncode == 0:
            return f"Memory Info:\n{result.stdout}"
        
        # Fallback to reading /proc/meminfo
        with open("/proc/meminfo", "r") as f:
            lines = f.readlines()[:3]
            return "Memory Info:\n" + "".join(lines)
    except Exception as e:
        return f"Memory Info: Error retrieving - {e}"


def _get_disk_info() -> str:
    """Get disk usage information."""
    try:
        result = subprocess.run(
            ["df", "-h", "/"],
            capture_output=True,
            text=True,
            timeout=2
        )
        
        if result.returncode == 0:
            return f"Disk Info:\n{result.stdout}"
        return "Disk Info: Unable to retrieve"
    except Exception as e:
        return f"Disk Info: Error retrieving - {e}"


def _get_process_info() -> str:
    """Get process count and top processes."""
    try:
        # Count processes
        result = subprocess.run(
            ["ps", "aux"],
            capture_output=True,
            text=True,
            timeout=2
        )
        
        if result.returncode == 0:
            process_count = len(result.stdout.strip().split("\n")) - 1  # Exclude header
            
            # Get top 5 processes by CPU
            top_result = subprocess.run(
                ["ps", "aux", "--sort=-%cpu"],
                capture_output=True,
                text=True,
                timeout=2
            )
            
            top_lines = top_result.stdout.strip().split("\n")[:6]  # Header + top 5
            
            return (
                f"Process Info:\n"
                f"  Total Processes: {process_count}\n"
                f"  Top 5 by CPU:\n" + "\n".join(top_lines)
            )
        
        return "Process Info: Unable to retrieve"
    except Exception as e:
        return f"Process Info: Error retrieving - {e}"


def tool_function(category: str = "all") -> str:
    """Get system statistics."""
    categories = ["cpu", "memory", "disk", "processes"] if category == "all" else [category]
    
    results = []
    for cat in categories:
        if cat == "cpu":
            results.append(_get_cpu_info())
        elif cat == "memory":
            results.append(_get_memory_info())
        elif cat == "disk":
            results.append(_get_disk_info())
        elif cat == "processes":
            results.append(_get_process_info())
    
    return "\n\n".join(results)
