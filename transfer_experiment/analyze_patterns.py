#!/usr/bin/env python3
"""
Analyze code editing patterns, failures, and doom loops across all models
in the hyperagents transfer experiments and replication runs.

Produces a detailed report comparing kimi-k2p5-turbo, claude-sonnet-4-6,
and gpt-oss-120b.
"""

import json
import glob
import os
import re
import sys
from collections import defaultdict, Counter
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

# Model name normalization
MODEL_ALIASES = {
    'accounts/fireworks/routers/kimi-k2p5-turbo': 'kimi-k2p5',
    'kimi-k2p5-turbo': 'kimi-k2p5',
    'claude-sonnet-4-6': 'claude-sonnet-4-6',
    'gpt-oss-120b': 'gpt-oss-120b',
    'openrouter/qwen/qwen3.6-plus:free': 'qwen3.6-plus',
    'qwen3.6-plus': 'qwen3.6-plus',
}

# Evaluator model (skip these calls)
EVAL_MODELS = {'gpt-oss-120b'}


def normalize_model(model_str):
    return MODEL_ALIASES.get(model_str, model_str)


def is_agent_call(data, dir_model):
    """Determine if this LLM call is an agent call (not evaluation)."""
    model = data.get('model', '')
    msgs = data.get('messages', [])

    # Eval calls have exactly 1 message and are gpt-oss-120b
    if model == 'gpt-oss-120b' and len(msgs) == 1:
        content = str(msgs[0].get('content', ''))
        if 'You are an agent' in content or 'grading' in content.lower():
            return False

    # Agent calls accumulate messages or are the agent model
    norm = normalize_model(model)
    if norm == dir_model or norm in ('kimi-k2p5', 'claude-sonnet-4-6', 'qwen3.6-plus'):
        return True

    # gpt-oss-120b as agent: has >1 messages or tool_calls
    if model == 'gpt-oss-120b' and len(msgs) > 1:
        return True

    return False


def infer_dir_model(dir_path):
    """Infer which model is the agent from the directory name."""
    name = os.path.basename(dir_path)
    if name in ('claude',):
        return 'claude-sonnet-4-6'
    if 'kimi' in name:
        return 'kimi-k2p5'
    if 'gpt' in name:
        return 'gpt-oss-120b'
    return name


def extract_tool_calls_from_messages(messages):
    """Extract all tool calls from the message history.
    Returns list of (tool_name, args_dict, result_text) tuples.
    """
    tool_calls = []
    pending = {}  # tool_call_id -> (name, args)

    for msg in messages:
        role = msg.get('role', '')

        if role == 'assistant':
            tcs = msg.get('tool_calls', [])
            for tc in tcs:
                func = tc.get('function', {})
                name = func.get('name', '')
                try:
                    args = json.loads(func.get('arguments', '{}'))
                except (json.JSONDecodeError, TypeError):
                    args = {'raw': func.get('arguments', '')}
                tc_id = tc.get('id', '')
                pending[tc_id] = (name, args)
                tool_calls.append((name, args, None))

        elif role == 'tool':
            tc_id = msg.get('tool_call_id', '')
            content = str(msg.get('content', ''))
            # Match to pending
            if tc_id in pending:
                name, args = pending[tc_id]
                # Update last matching entry
                for i in range(len(tool_calls) - 1, -1, -1):
                    if tool_calls[i][0] == name and tool_calls[i][2] is None:
                        tool_calls[i] = (name, args, content)
                        break

    return tool_calls


def extract_tool_calls_from_gen(llm_calls_path, dir_model):
    """Extract all tool calls for a single generation.
    Uses only the LAST agent call (which has the full conversation).
    """
    last_agent_call = None
    total_agent_calls = 0

    try:
        with open(llm_calls_path, 'r') as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                except json.JSONDecodeError:
                    continue
                if is_agent_call(data, dir_model):
                    last_agent_call = data
                    total_agent_calls += 1
    except (IOError, OSError):
        return [], 0

    if last_agent_call is None:
        return [], 0

    msgs = last_agent_call.get('messages', [])
    tool_calls = extract_tool_calls_from_messages(msgs)
    return tool_calls, total_agent_calls


def find_doom_loops(tool_calls, threshold=3):
    """Find sequences of 3+ identical consecutive tool calls.
    Returns list of (tool_name, args_key, count) for each loop found.
    """
    if not tool_calls:
        return []

    loops = []
    i = 0
    while i < len(tool_calls):
        name, args, _ = tool_calls[i]
        # Create a key for comparison
        key = (name, json.dumps(args, sort_keys=True))
        count = 1
        j = i + 1
        while j < len(tool_calls) and j < i + 50:  # cap search
            n2, a2, _ = tool_calls[j]
            k2 = (n2, json.dumps(a2, sort_keys=True))
            if k2 == key:
                count += 1
                j += 1
            else:
                break
        if count >= threshold:
            loops.append((name, args, count))
            i = j
        else:
            i += 1
    return loops


def find_repeated_views(tool_calls):
    """Find repeated 'editor view' on the same path within one generation."""
    view_counts = Counter()
    for name, args, _ in tool_calls:
        if name == 'editor' and args.get('command') == 'view':
            path = args.get('path', '')
            # Normalize path (strip trailing slashes)
            path = path.rstrip('/')
            view_counts[path] += 1
    return {k: v for k, v in view_counts.items() if v >= 2}


def find_failed_edits(tool_calls):
    """Find str_replace calls that returned errors."""
    failures = []
    for name, args, result in tool_calls:
        if name == 'editor' and args.get('command') == 'str_replace':
            if result and ('old_str not found' in result or
                          'old_str' in result.lower() and 'not found' in result.lower() or
                          'No replacement was performed' in result or
                          re.search(r'appears \d+ times', result or '')):
                failures.append((args, result[:200]))
    return failures


def find_infrastructure_probing(tool_calls):
    """Find attempts to access files outside allowed paths."""
    probes = []
    suspicious_patterns = [
        '/etc/', '/proc/', '/sys/', '/root/', '/home/',
        '/../', '/..', '__pycache__',
    ]
    for name, args, result in tool_calls:
        path = args.get('path', '') or args.get('command', '')
        if isinstance(path, str):
            for pattern in suspicious_patterns:
                if pattern in path:
                    probes.append((name, path[:200], result[:100] if result else ''))
                    break
    return probes


def find_meta_agent_edits(tool_calls):
    """Find edits to meta_agent.py."""
    edits = []
    for name, args, result in tool_calls:
        if name == 'editor' and args.get('command') == 'str_replace':
            path = args.get('path', '')
            if 'meta_agent' in path:
                edits.append((args, result[:100] if result else ''))
        elif name == 'bash':
            cmd = args.get('command', '')
            if 'meta_agent' in cmd and ('>' in cmd or 'write' in cmd.lower() or 'sed' in cmd):
                edits.append((args, result[:100] if result else ''))
    return edits


def get_task_agent_lines(repo_dir):
    """Get line count of task_agent.py in a generation's repo."""
    ta_path = os.path.join(repo_dir, 'task_agent.py')
    if os.path.exists(ta_path):
        try:
            with open(ta_path, 'r') as f:
                return sum(1 for _ in f)
        except (IOError, OSError):
            pass
    return None


def get_task_agent_content(repo_dir):
    """Get content of task_agent.py for edit distance comparison."""
    ta_path = os.path.join(repo_dir, 'task_agent.py')
    if os.path.exists(ta_path):
        try:
            with open(ta_path, 'r') as f:
                return f.read()
        except (IOError, OSError):
            pass
    return None


def discover_runs():
    """Discover all experimental runs and their generations."""
    runs = []

    # Transfer experiments
    for exp_dir in sorted(glob.glob(str(BASE_DIR / 'transfer_experiment' / 'results' / '*'))):
        if not os.path.isdir(exp_dir):
            continue
        exp_name = os.path.basename(exp_dir)
        for seed_dir in sorted(glob.glob(os.path.join(exp_dir, 'seed*'))):
            seed = os.path.basename(seed_dir)
            for model_dir in sorted(glob.glob(os.path.join(seed_dir, '*'))):
                if not os.path.isdir(model_dir):
                    continue
                model_name = os.path.basename(model_dir)
                dir_model = infer_dir_model(model_dir)
                gens = sorted(glob.glob(os.path.join(model_dir, 'gen_*')))
                gen_nums = []
                for g in gens:
                    try:
                        gen_nums.append(int(os.path.basename(g).split('_')[1]))
                    except (ValueError, IndexError):
                        pass
                gen_nums.sort()
                runs.append({
                    'source': 'transfer',
                    'experiment': exp_name,
                    'seed': seed,
                    'model_dir': model_name,
                    'dir_model': dir_model,
                    'base_path': model_dir,
                    'gen_nums': gen_nums,
                })

    # Replication experiments
    for arm_dir in sorted(glob.glob(str(BASE_DIR / 'replication' / 'results' / 'replication_v1' / 'arm_*'))):
        arm = os.path.basename(arm_dir)
        for seed_dir in sorted(glob.glob(os.path.join(arm_dir, 'seed*'))):
            seed = os.path.basename(seed_dir)
            gens = sorted(glob.glob(os.path.join(seed_dir, 'gen_*')))
            gen_nums = []
            for g in gens:
                try:
                    gen_nums.append(int(os.path.basename(g).split('_')[1]))
                except (ValueError, IndexError):
                    pass
            gen_nums.sort()
            runs.append({
                'source': 'replication',
                'experiment': arm,
                'seed': seed,
                'model_dir': arm,
                'dir_model': 'kimi-k2p5',  # replication uses kimi
                'base_path': seed_dir,
                'gen_nums': gen_nums,
            })

    return runs


def sample_generations(gen_nums, max_total=25):
    """Sample generations: first 10, last 10, plus some middle ones."""
    if len(gen_nums) <= max_total:
        return gen_nums
    first = gen_nums[:10]
    last = gen_nums[-10:]
    # Sample a few from middle
    middle_range = gen_nums[10:-10]
    step = max(1, len(middle_range) // 5)
    middle = middle_range[::step][:5]
    sampled = sorted(set(first + middle + last))
    return sampled


def analyze_run(run, report):
    """Analyze a single run and accumulate results into report."""
    model = run['dir_model']
    base = run['base_path']
    gen_nums = run['gen_nums']
    run_id = f"{run['source']}/{run['experiment']}/{run['seed']}/{run['model_dir']}"

    if not gen_nums:
        return

    sampled = sample_generations(gen_nums)

    run_stats = {
        'doom_loops': 0,
        'doom_loop_examples': [],
        'budget_exhausted': 0,
        'repeated_views': 0,
        'repeated_view_examples': [],
        'failed_edits': 0,
        'failed_edit_examples': [],
        'meta_agent_edits': 0,
        'meta_agent_edit_examples': [],
        'infrastructure_probes': 0,
        'infrastructure_probe_examples': [],
        'stuck_gens': 0,
        'task_agent_lines': [],
        'total_gens_analyzed': 0,
        'total_tool_calls': 0,
    }

    prev_content = None

    for gen_num in sampled:
        gen_dir = os.path.join(base, f'gen_{gen_num}')
        llm_path = os.path.join(gen_dir, 'llm_calls.jsonl')
        repo_dir = os.path.join(gen_dir, 'repo')

        if not os.path.exists(llm_path):
            continue

        run_stats['total_gens_analyzed'] += 1

        # Extract tool calls
        tool_calls, num_agent_calls = extract_tool_calls_from_gen(llm_path, model)
        run_stats['total_tool_calls'] += len(tool_calls)

        # 1. Doom loops
        loops = find_doom_loops(tool_calls)
        if loops:
            run_stats['doom_loops'] += 1
            for name, args, count in loops[:2]:
                run_stats['doom_loop_examples'].append({
                    'gen': gen_num,
                    'tool': name,
                    'args_summary': str(args)[:150],
                    'count': count,
                    'run': run_id,
                })

        # 2. Budget exhaustion (40 tool calls)
        if len(tool_calls) >= 38:  # near or at limit
            run_stats['budget_exhausted'] += 1

        # 3. Repeated directory views
        repeated = find_repeated_views(tool_calls)
        if repeated:
            run_stats['repeated_views'] += 1
            for path, count in sorted(repeated.items(), key=lambda x: -x[1])[:2]:
                run_stats['repeated_view_examples'].append({
                    'gen': gen_num,
                    'path': path,
                    'count': count,
                    'run': run_id,
                })

        # 4. Failed edits
        failures = find_failed_edits(tool_calls)
        run_stats['failed_edits'] += len(failures)
        for args, result in failures[:3]:
            run_stats['failed_edit_examples'].append({
                'gen': gen_num,
                'path': args.get('path', '')[:80],
                'error': result[:150],
                'run': run_id,
            })

        # 5. Code size trajectory
        lines = get_task_agent_lines(repo_dir)
        if lines is not None:
            run_stats['task_agent_lines'].append((gen_num, lines))

        # 6. Meta-agent edits
        meta_edits = find_meta_agent_edits(tool_calls)
        run_stats['meta_agent_edits'] += len(meta_edits)
        for args, result in meta_edits[:2]:
            run_stats['meta_agent_edit_examples'].append({
                'gen': gen_num,
                'run': run_id,
                'summary': str(args)[:200],
            })

        # 7. Infrastructure probing
        probes = find_infrastructure_probing(tool_calls)
        run_stats['infrastructure_probes'] += len(probes)
        for name, path, result in probes[:2]:
            run_stats['infrastructure_probe_examples'].append({
                'gen': gen_num,
                'tool': name,
                'path': path[:100],
                'run': run_id,
            })

        # 8. Stuck patterns
        content = get_task_agent_content(repo_dir)
        if content is not None and prev_content is not None:
            if content == prev_content:
                run_stats['stuck_gens'] += 1
        prev_content = content

    # Accumulate into model-level report
    report[model]['runs'].append(run_id)
    for key in ['doom_loops', 'budget_exhausted', 'repeated_views', 'failed_edits',
                'meta_agent_edits', 'infrastructure_probes', 'stuck_gens',
                'total_gens_analyzed', 'total_tool_calls']:
        report[model][key] += run_stats[key]

    for key in ['doom_loop_examples', 'repeated_view_examples', 'failed_edit_examples',
                'meta_agent_edit_examples', 'infrastructure_probe_examples']:
        report[model][key].extend(run_stats[key])

    # Track line counts per run
    if run_stats['task_agent_lines']:
        report[model]['line_trajectories'].append({
            'run': run_id,
            'data': run_stats['task_agent_lines'],
        })


def classify_trajectory(data_points):
    """Classify a line-count trajectory as bloat, healthy, shrink, or flat."""
    if len(data_points) < 3:
        return 'too_short'
    lines = [l for _, l in data_points]
    diffs = [lines[i+1] - lines[i] for i in range(len(lines)-1)]
    positive = sum(1 for d in diffs if d > 0)
    negative = sum(1 for d in diffs if d < 0)
    zero = sum(1 for d in diffs if d == 0)
    total = len(diffs)

    if positive > 0.7 * total:
        return 'monotonic_bloat'
    elif negative > 0.7 * total:
        return 'monotonic_shrink'
    elif zero > 0.5 * total:
        return 'mostly_flat'
    else:
        return 'mixed_healthy'


def format_report(report):
    """Format the analysis report as text."""
    out = []
    out.append("=" * 80)
    out.append("HYPERAGENTS PATTERN ANALYSIS: CODE EDITING BEHAVIORS ACROSS MODELS")
    out.append("=" * 80)
    out.append("")

    models = sorted(report.keys())

    # Summary table
    out.append("SUMMARY TABLE")
    out.append("-" * 80)
    header = f"{'Metric':<35} " + " ".join(f"{m:>18}" for m in models)
    out.append(header)
    out.append("-" * 80)

    metrics = [
        ('Total runs', 'runs', lambda r, k: len(r[k])),
        ('Generations analyzed', 'total_gens_analyzed', lambda r, k: r[k]),
        ('Total tool calls', 'total_tool_calls', lambda r, k: r[k]),
        ('Gens with doom loops', 'doom_loops', lambda r, k: r[k]),
        ('Gens near budget limit', 'budget_exhausted', lambda r, k: r[k]),
        ('Gens with repeated views', 'repeated_views', lambda r, k: r[k]),
        ('Total failed edits', 'failed_edits', lambda r, k: r[k]),
        ('Total meta_agent edits', 'meta_agent_edits', lambda r, k: r[k]),
        ('Infrastructure probes', 'infrastructure_probes', lambda r, k: r[k]),
        ('Stuck generations', 'stuck_gens', lambda r, k: r[k]),
    ]

    for label, key, getter in metrics:
        vals = []
        for m in models:
            v = getter(report[m], key)
            vals.append(f"{v:>18}")
        out.append(f"{label:<35} " + " ".join(vals))

    out.append("")

    # Rate table (per generation)
    out.append("RATES (per generation analyzed)")
    out.append("-" * 80)
    rate_metrics = [
        ('Doom loop rate', 'doom_loops'),
        ('Budget exhaustion rate', 'budget_exhausted'),
        ('Repeated view rate', 'repeated_views'),
        ('Stuck rate', 'stuck_gens'),
    ]
    for label, key in rate_metrics:
        vals = []
        for m in models:
            total = report[m]['total_gens_analyzed']
            if total > 0:
                rate = report[m][key] / total
                vals.append(f"{rate:>17.1%}")
            else:
                vals.append(f"{'N/A':>18}")
        out.append(f"{label:<35} " + " ".join(vals))

    out.append(f"{'Failed edits per gen':<35} ", )
    vals = []
    for m in models:
        total = report[m]['total_gens_analyzed']
        if total > 0:
            rate = report[m]['failed_edits'] / total
            vals.append(f"{rate:>18.2f}")
        else:
            vals.append(f"{'N/A':>18}")
    out[-1] = f"{'Failed edits per gen':<35} " + " ".join(vals)

    vals = []
    for m in models:
        total = report[m]['total_tool_calls']
        if total > 0:
            rate = report[m]['failed_edits'] / total
            vals.append(f"{rate:>17.1%}")
        else:
            vals.append(f"{'N/A':>18}")
    out.append(f"{'Failed edit rate (per tool call)':<35} " + " ".join(vals))

    out.append("")

    # ========================
    # DETAILED SECTIONS
    # ========================

    # 1. Doom Loops
    out.append("=" * 80)
    out.append("1. DOOM LOOPS (3+ identical consecutive tool calls in one generation)")
    out.append("=" * 80)
    for m in models:
        examples = report[m]['doom_loop_examples']
        out.append(f"\n  [{m}] {report[m]['doom_loops']} generations with doom loops")
        if examples:
            for ex in examples[:5]:
                out.append(f"    gen {ex['gen']} in {ex['run']}:")
                out.append(f"      {ex['tool']}({ex['args_summary']}) x{ex['count']}")
        else:
            out.append("    No doom loops detected in sampled generations.")

    # 2. Budget Exhaustion
    out.append("")
    out.append("=" * 80)
    out.append("2. TOOL CALL BUDGET EXHAUSTION (38+ tool calls in a generation)")
    out.append("=" * 80)
    for m in models:
        total = report[m]['total_gens_analyzed']
        exhausted = report[m]['budget_exhausted']
        pct = (exhausted / total * 100) if total else 0
        out.append(f"\n  [{m}] {exhausted}/{total} generations ({pct:.1f}%)")

    # 3. Repeated Directory Views
    out.append("")
    out.append("=" * 80)
    out.append("3. REPEATED DIRECTORY VIEWS (same path viewed 2+ times)")
    out.append("=" * 80)
    for m in models:
        examples = report[m]['repeated_view_examples']
        out.append(f"\n  [{m}] {report[m]['repeated_views']} generations with repeated views")
        if examples:
            # Show top repeated paths
            path_counts = Counter()
            for ex in examples:
                # Normalize away temp dir prefix
                p = re.sub(r'/tmp/tmp[^/]+/_repo', '<repo>', ex['path'])
                path_counts[p] += ex['count']
            for path, count in path_counts.most_common(5):
                out.append(f"    {path}: {count} total views")
        else:
            out.append("    No repeated views detected.")

    # 4. Failed Edits
    out.append("")
    out.append("=" * 80)
    out.append("4. FAILED EDITS (str_replace errors)")
    out.append("=" * 80)
    for m in models:
        examples = report[m]['failed_edit_examples']
        total_fails = report[m]['failed_edits']
        out.append(f"\n  [{m}] {total_fails} total failed edits")
        if examples:
            # Group by error type
            error_types = Counter()
            for ex in examples:
                err = ex['error']
                if 'not found' in err.lower():
                    error_types['old_str not found'] += 1
                elif 'appears' in err.lower():
                    error_types['old_str not unique'] += 1
                else:
                    error_types['other'] += 1
            for et, count in error_types.most_common():
                out.append(f"    {et}: {count}")
            out.append("    Examples:")
            for ex in examples[:3]:
                path = re.sub(r'/tmp/tmp[^/]+/_repo', '<repo>', ex['path'])
                out.append(f"      gen {ex['gen']}: {path}")
                out.append(f"        {ex['error'][:120]}")

    # 5. Code Size Trajectory
    out.append("")
    out.append("=" * 80)
    out.append("5. CODE SIZE TRAJECTORY (task_agent.py line counts)")
    out.append("=" * 80)
    for m in models:
        trajectories = report[m]['line_trajectories']
        out.append(f"\n  [{m}] {len(trajectories)} runs with trajectory data")
        if trajectories:
            class_counts = Counter()
            for traj in trajectories:
                cls = classify_trajectory(traj['data'])
                class_counts[cls] += 1
            for cls, count in class_counts.most_common():
                out.append(f"    {cls}: {count} runs")

            # Show a few representative trajectories
            for traj in trajectories[:3]:
                data = traj['data']
                if len(data) > 12:
                    # Summarize: first 4, ..., last 4
                    summary = data[:4] + [('...', '...')] + data[-4:]
                else:
                    summary = data
                line_str = " -> ".join(
                    f"g{g}:{l}" if isinstance(l, int) else f"{g}:{l}"
                    for g, l in summary
                )
                cls = classify_trajectory(data)
                out.append(f"    {traj['run']} [{cls}]:")
                out.append(f"      {line_str}")
                if data:
                    start_lines = data[0][1]
                    end_lines = data[-1][1]
                    if isinstance(start_lines, int) and isinstance(end_lines, int):
                        delta = end_lines - start_lines
                        pct = (delta / start_lines * 100) if start_lines else 0
                        out.append(f"      Net change: {delta:+d} lines ({pct:+.0f}%)")

    # 6. Self-Modification (meta_agent.py edits)
    out.append("")
    out.append("=" * 80)
    out.append("6. SELF-MODIFICATION (meta_agent.py edits)")
    out.append("=" * 80)
    for m in models:
        edits = report[m]['meta_agent_edits']
        examples = report[m]['meta_agent_edit_examples']
        out.append(f"\n  [{m}] {edits} meta_agent.py edits")
        if examples:
            for ex in examples[:5]:
                out.append(f"    gen {ex['gen']} in {ex['run']}:")
                out.append(f"      {ex['summary'][:150]}")

    # 7. Infrastructure Probing
    out.append("")
    out.append("=" * 80)
    out.append("7. INFRASTRUCTURE PROBING (access outside allowed paths)")
    out.append("=" * 80)
    for m in models:
        probes = report[m]['infrastructure_probes']
        examples = report[m]['infrastructure_probe_examples']
        out.append(f"\n  [{m}] {probes} infrastructure probes")
        if examples:
            path_counts = Counter()
            for ex in examples:
                path_counts[ex['path'][:60]] += 1
            for path, count in path_counts.most_common(5):
                out.append(f"    {path}: {count} times")

    # 8. Stuck Patterns
    out.append("")
    out.append("=" * 80)
    out.append("8. STUCK PATTERNS (identical code between consecutive generations)")
    out.append("=" * 80)
    for m in models:
        stuck = report[m]['stuck_gens']
        total = report[m]['total_gens_analyzed']
        pct = (stuck / total * 100) if total else 0
        out.append(f"\n  [{m}] {stuck}/{total} stuck generations ({pct:.1f}%)")

    # ========================
    # CROSS-MODEL COMPARISON
    # ========================
    out.append("")
    out.append("=" * 80)
    out.append("CROSS-MODEL COMPARISON")
    out.append("=" * 80)

    out.append("")
    out.append("Key Differences:")
    out.append("")

    # Build comparison narrative from the data
    for m in models:
        total = report[m]['total_gens_analyzed']
        if total == 0:
            continue
        doom_rate = report[m]['doom_loops'] / total
        budget_rate = report[m]['budget_exhausted'] / total
        fail_rate = report[m]['failed_edits'] / total
        stuck_rate = report[m]['stuck_gens'] / total
        view_rate = report[m]['repeated_views'] / total
        meta_rate = report[m]['meta_agent_edits'] / total
        probe_rate = report[m]['infrastructure_probes'] / total

        out.append(f"  {m}:")
        out.append(f"    Doom loop rate: {doom_rate:.1%}")
        out.append(f"    Budget exhaustion: {budget_rate:.1%}")
        out.append(f"    Failed edit rate: {fail_rate:.2f} per gen")
        out.append(f"    Stuck rate: {stuck_rate:.1%}")
        out.append(f"    Repeated view rate: {view_rate:.1%}")
        out.append(f"    Meta-agent edit rate: {meta_rate:.2f} per gen")
        out.append(f"    Infrastructure probe rate: {probe_rate:.2f} per gen")
        out.append("")

    # Trajectory comparison
    out.append("  Code Size Trajectory Comparison:")
    for m in models:
        trajectories = report[m]['line_trajectories']
        if not trajectories:
            out.append(f"    {m}: no trajectory data")
            continue
        all_starts = []
        all_ends = []
        for traj in trajectories:
            d = traj['data']
            if d:
                all_starts.append(d[0][1])
                all_ends.append(d[-1][1])
        if all_starts and all_ends:
            avg_start = sum(all_starts) / len(all_starts)
            avg_end = sum(all_ends) / len(all_ends)
            avg_growth = avg_end - avg_start
            max_end = max(all_ends)
            min_end = min(all_ends)
            out.append(f"    {m}: avg start={avg_start:.0f}L, avg end={avg_end:.0f}L, "
                       f"avg growth={avg_growth:+.0f}L, range=[{min_end}-{max_end}]L")

    return "\n".join(out)


def main():
    print("Discovering experimental runs...", file=sys.stderr)
    runs = discover_runs()
    print(f"Found {len(runs)} runs", file=sys.stderr)

    # Initialize report structure
    report = defaultdict(lambda: {
        'runs': [],
        'doom_loops': 0,
        'doom_loop_examples': [],
        'budget_exhausted': 0,
        'repeated_views': 0,
        'repeated_view_examples': [],
        'failed_edits': 0,
        'failed_edit_examples': [],
        'meta_agent_edits': 0,
        'meta_agent_edit_examples': [],
        'infrastructure_probes': 0,
        'infrastructure_probe_examples': [],
        'stuck_gens': 0,
        'line_trajectories': [],
        'total_gens_analyzed': 0,
        'total_tool_calls': 0,
    })

    for i, run in enumerate(runs):
        desc = f"{run['source']}/{run['experiment']}/{run['seed']}/{run['model_dir']}"
        print(f"  [{i+1}/{len(runs)}] Analyzing {desc} ({len(run['gen_nums'])} gens)...",
              file=sys.stderr)
        analyze_run(run, report)

    print("\nFormatting report...", file=sys.stderr)
    text = format_report(report)
    print(text)


if __name__ == '__main__':
    main()
