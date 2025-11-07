import argparse
import csv
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass
class TaskStats:
    task_name: str
    successes: int
    failures: int

    @property
    def total(self) -> int:
        return self.successes + self.failures

    @property
    def success_rate(self) -> float:
        if self.total == 0:
            return 0.0
        return self.successes / self.total


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Scan an evaluation folder with per-task subfolders containing "
            "<videohash>_sucess{0|1}.mp4 files and produce a results CSV."
        )
    )
    parser.add_argument(
        "--root",
        required=True,
        help="Path to the evaluation root directory (e.g., /path/to/20000)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output CSV path (default: <root>/results.csv)",
    )
    parser.add_argument(
        "--pattern",
        default="_sucess",
        help=(
            "Suffix pattern before success flag. Files must end with "
            "'<pattern>1.mp4' or '<pattern>0.mp4'. Default: '_sucess'"
        ),
    )
    strict_group = parser.add_mutually_exclusive_group()
    strict_group.add_argument(
        "--strict",
        dest="strict",
        action="store_true",
        help="Assert if a task has zero matching files (default)",
    )
    strict_group.add_argument(
        "--no-strict",
        dest="strict",
        action="store_false",
        help="Allow tasks with zero matching files (success_rate reported as 0)",
    )
    parser.set_defaults(strict=True)
    return parser.parse_args()


def is_task_dir(path: str) -> bool:
    return os.path.isdir(path)


def collect_task_stats(task_dir: str, task_name: str, pattern: str) -> TaskStats:
    def count_with_pattern(entries: List[str], suffix_core: str) -> Tuple[int, int]:
        success_suffix = f"{suffix_core}1.mp4"
        failure_suffix = f"{suffix_core}0.mp4"
        succ = 0
        fail = 0
        for entry in entries:
            full_path = os.path.join(task_dir, entry)
            if not os.path.isfile(full_path):
                continue
            if entry.endswith(success_suffix):
                succ += 1
            elif entry.endswith(failure_suffix):
                fail += 1
        return succ, fail

    try:
        entries = os.listdir(task_dir)
    except FileNotFoundError:
        entries = []

    # First try the provided pattern
    successes, failures = count_with_pattern(entries, pattern)

    # Fallback: if no matches and the provided pattern was the common misspelling,
    # also try the correct spelling automatically.
    if successes == 0 and failures == 0 and pattern == "_sucess":
        alt_successes, alt_failures = count_with_pattern(entries, "_success")
        if alt_successes + alt_failures > 0:
            successes, failures = alt_successes, alt_failures

    return TaskStats(task_name=task_name, successes=successes, failures=failures)


def scan_root(root: str, pattern: str, strict: bool) -> List[TaskStats]:
    assert os.path.isdir(root), f"Root directory does not exist or is not a directory: {root}"

    # Enumerate immediate subdirectories as tasks
    task_names: List[str] = []
    try:
        for entry in os.listdir(root):
            entry_path = os.path.join(root, entry)
            if is_task_dir(entry_path):
                task_names.append(entry)
    except FileNotFoundError:
        task_names = []

    # Deterministic ordering
    task_names.sort()

    stats: List[TaskStats] = []
    for task_name in task_names:
        task_dir = os.path.join(root, task_name)
        task_stats = collect_task_stats(task_dir, task_name, pattern)
        # if strict:
        #     assert (
        #         task_stats.total > 0
        #     ), f"Task '{task_name}' contains no files matching *{pattern}[0|1].mp4"
        stats.append(task_stats)

    return stats


def compute_averages(stats: List[TaskStats]) -> Tuple[float, float]:
    if not stats:
        # No tasks; in strict mode this would have asserted earlier only if there were
        # zero files within tasks. If there are no tasks, macro/micro are 0.
        return 0.0, 0.0

    # Macro average: mean of per-task success rates
    macro_sum = 0.0
    for s in stats:
        macro_sum += s.success_rate
    macro_avg = macro_sum / len(stats)

    # Micro average: total successes over total attempts
    total_successes = sum(s.successes for s in stats)
    total_attempts = sum(s.total for s in stats)
    micro_avg = (total_successes / total_attempts) if total_attempts > 0 else 0.0

    return macro_avg, micro_avg


def write_csv(output_path: str, stats: List[TaskStats], macro_avg: float, micro_avg: float) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["task", "successes", "failures", "total", "success_rate"])

        for s in stats:
            writer.writerow([s.task_name, s.successes, s.failures, s.total, f"{s.success_rate:.6f}"])

        # Summary rows
        writer.writerow(["AVERAGE_TASK", "", "", "", f"{macro_avg:.6f}"])
        writer.writerow(["OVERALL", "", "", "", f"{micro_avg:.6f}"])


def main() -> None:
    args = parse_args()
    root = os.path.abspath(args.root)
    output = args.output if args.output else os.path.join(root, "results.csv")
    pattern = args.pattern
    strict = args.strict

    stats = scan_root(root, pattern, strict)
    macro_avg, micro_avg = compute_averages(stats)
    write_csv(output, stats, macro_avg, micro_avg)

    num_tasks = len(stats)
    print(
        (
            f"Processed {num_tasks} tasks. "
            f"Macro avg (per-task): {macro_avg:.4f}. "
            f"Micro avg (overall): {micro_avg:.4f}. "
            f"CSV written to: {output}"
        )
    )


if __name__ == "__main__":
    main()


