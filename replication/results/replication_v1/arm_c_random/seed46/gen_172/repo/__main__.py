"""
Main entry point for running the agent as a module.

Allows running with: python -m repo
"""

from __future__ import annotations

import argparse
import json
import logging
import sys

from task_agent import TaskAgent
from meta_agent import MetaAgent

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def run_task_agent(inputs_path: str, output_path: str | None = None) -> None:
    """Run the task agent on a single input file."""
    with open(inputs_path) as f:
        inputs = json.load(f)
    
    agent = TaskAgent()
    prediction, msg_history = agent.forward(inputs)
    
    result = {
        "prediction": prediction,
        "msg_history": msg_history,
    }
    
    if output_path:
        with open(output_path, "w") as f:
            json.dump(result, f, indent=2)
        logger.info(f"Result written to {output_path}")
    else:
        print(json.dumps(result, indent=2))


def run_meta_agent(repo_path: str, eval_path: str | None = None, iterations: int | None = None) -> None:
    """Run the meta agent to modify the codebase."""
    agent = MetaAgent()
    msg_history = agent.forward(repo_path, eval_path or "", iterations)
    
    # Print summary
    num_messages = len(msg_history)
    logger.info(f"Meta agent completed with {num_messages} messages in history")
    
    # Check for any tool calls in the history
    tool_calls = 0
    for msg in msg_history:
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            tool_calls += len(msg.get("tool_calls", []))
    
    logger.info(f"Total tool calls made: {tool_calls}")


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Agent module runner")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Task agent command
    task_parser = subparsers.add_parser("task", help="Run task agent")
    task_parser.add_argument("inputs", help="Path to inputs JSON file")
    task_parser.add_argument("-o", "--output", help="Path to output JSON file (optional)")
    
    # Meta agent command
    meta_parser = subparsers.add_parser("meta", help="Run meta agent")
    meta_parser.add_argument("repo_path", help="Path to repository to modify")
    meta_parser.add_argument("-e", "--eval-path", help="Path to evaluation results (optional)")
    meta_parser.add_argument("-i", "--iterations", type=int, help="Remaining iterations (optional)")
    
    args = parser.parse_args()
    
    if args.command == "task":
        run_task_agent(args.inputs, args.output)
    elif args.command == "meta":
        run_meta_agent(args.repo_path, args.eval_path, args.iterations)
    else:
        parser.print_help()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
