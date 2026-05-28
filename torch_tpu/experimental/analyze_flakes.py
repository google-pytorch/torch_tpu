# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from datetime import datetime, timedelta, timezone
import json
import subprocess
import sys


def run_cmd(args):
  result = subprocess.run(args, capture_output=True, text=True)
  if result.returncode != 0:
    print(f"Error running {' '.join(args)}: {result.stderr}", file=sys.stderr)
    return None
  return result.stdout


def drill_down_target(target):
  print(f"  Drilling down into {target}...")
  executions_out = run_cmd([
      "/google/bin/releases/flakes-cli/flakes-cli",
      "executions",
      target,
      "--output_records_limit",
      "10",
  ])
  if not executions_out:
    return [], []

  try:
    executions = json.loads(executions_out)
  except json.JSONDecodeError:
    return [], []

  failed_invocations = []
  timed_out_invocations = []
  inv_statuses = {}
  for exec_item in executions:
    exec_id = exec_item.get("execution_id", {})
    if exec_id.get("type") == "sponge":
      inv_id = exec_id.get("invocation_id")
      status = exec_item.get("status")
      inv_statuses[inv_id] = status
      if status == "FAILED":
        if inv_id not in failed_invocations:
          failed_invocations.append(inv_id)
      elif status == "TIMED_OUT":
        if inv_id not in timed_out_invocations:
          timed_out_invocations.append(inv_id)

  invocations_to_check = failed_invocations[:3]
  if len(invocations_to_check) < 3:
    invocations_to_check.extend(
        timed_out_invocations[: 3 - len(invocations_to_check)]
    )

  test_failures = {}
  links = []
  for inv_id in invocations_to_check:
    status = inv_statuses.get(inv_id, "UNKNOWN")
    links.append({"id": inv_id, "status": status})

    failures_out = run_cmd([
        "/google/bin/releases/gemini-agents-sponge-cli/sponge",
        "failures",
        f"--invocation_id={inv_id}",
        f"--target={target}",
        "--json",
    ])
    if not failures_out:
      continue

    try:
      failures = json.loads(failures_out)
    except json.JSONDecodeError:
      continue

    for failure in failures:
      if failure.get("type") == "TEST":
        name = failure.get("name")
        if name and not name.startswith("Action:"):
          test_failures[name] = test_failures.get(name, 0) + 1
        elif name and name.startswith("Action:"):
          msg = failure.get("message", "Unknown error")
          if "terminated with signal" in msg:
            msg = msg.split("terminated with signal")[1].strip()
          elif "failed with status" in msg:
            msg = msg.split("failed with status")[1].strip()
          shard_name = name.split(" (")[0].replace("Action: ", "")
          test_failures[f"[{shard_name}] {msg}"] = (
              test_failures.get(f"[{shard_name}] {msg}", 0) + 1
          )

  sorted_failures = sorted(
      test_failures.items(), key=lambda x: x[1], reverse=True
  )
  return sorted_failures, links


def categorize_issues(report_data, standalone_test_cases):
  categories = {
      "Timeout / Infrastructure Hangs": {
          "desc": (
              "Targets primarily failing due to exceeding execution time"
              " limits, TPU allocation timeouts, or hangs (manifested as shard"
              " TIMED_OUT or INCOMPLETE)."
          ),
          "targets": [],
      },
      "Distributed & Collective Communication Failures": {
          "desc": (
              "Genuinely flaky test cases where PyTorch multiprocessing,"
              " process spawn, or TPU collective operations failed (often with"
              " SIGABRT)."
          ),
          "targets": [],
          "test_cases": [],
      },
      "Model & Functional Equivalence Failures": {
          "desc": (
              "Flakiness in functional correctness, such as numerical"
              " equivalence checks between different implementations."
          ),
          "targets": [],
          "test_cases": [],
      },
      "Sanitizer & Memory Violations": {
          "desc": (
              "Failures detected under sanitizer configurations (e.g.,"
              " MemorySanitizer), indicating potential uninitialized memory"
              " reads or memory safety issues."
          ),
          "targets": [],
          "test_cases": [],
      },
  }

  for item in report_data:
    target = item["target"]
    drill_down = item["drill_down"]
    avg_14 = item["avg_14"]

    if avg_14 <= 0.001:
      continue

    has_msan = False
    has_dist = False
    has_model = False
    has_actual_tests = False

    for name, count in drill_down:
      if not name.startswith("[test_shard"):
        has_actual_tests = True
        if "MemorySanitizer" in name:
          has_msan = True
        elif any(
            x in name
            for x in [
                "all_reduce",
                "all_gather",
                "send_recv",
                "collective",
                "dist",
            ]
        ):
          has_dist = True
        elif "equivalence" in name or "model" in name or "accuracy" in name:
          has_model = True
        else:
          if "distributed" in target:
            has_dist = True
          else:
            has_model = True

    if has_msan:
      categories["Sanitizer & Memory Violations"]["targets"].append(target)
      for tc in standalone_test_cases:
        if tc["target"] == target and "MemorySanitizer" in tc["test_case"]:
          categories["Sanitizer & Memory Violations"]["test_cases"].append(tc)
    elif has_dist:
      categories["Distributed & Collective Communication Failures"][
          "targets"
      ].append(target)
      for tc in standalone_test_cases:
        if tc["target"] == target:
          categories["Distributed & Collective Communication Failures"][
              "test_cases"
          ].append(tc)
    elif has_model:
      categories["Model & Functional Equivalence Failures"]["targets"].append(
          target
      )
      for tc in standalone_test_cases:
        if tc["target"] == target:
          categories["Model & Functional Equivalence Failures"][
              "test_cases"
          ].append(tc)
    else:
      categories["Timeout / Infrastructure Hangs"]["targets"].append(target)

  return categories


def main():
  project = "third_party.py.torch_tpu"
  threshold = 0.001  # 0.1% default
  out_path = None

  if len(sys.argv) > 1:
    out_path = sys.argv[1]
  if len(sys.argv) > 2:
    try:
      threshold = float(sys.argv[2])
    except ValueError:
      print(
          f"Invalid threshold value: {sys.argv[2]}, using default {threshold}",
          file=sys.stderr,
      )

  print(f"Using flakiness threshold: {threshold:.2%}")
  print(f"Fetching target list for {project}...")
  target_list_out = run_cmd([
      "/google/bin/releases/flakes-cli/flakes-cli",
      "target-list",
      "tap-project",
      project,
  ])
  if not target_list_out:
    return

  try:
    targets = json.loads(target_list_out)
  except json.JSONDecodeError as e:
    print(f"Failed to decode target-list JSON: {e}", file=sys.stderr)
    return

  report_data = []

  now = datetime.now(timezone.utc)
  cutoff_7_days = now - timedelta(days=7)
  cutoff_14_days = now - timedelta(days=14)

  for item in targets:
    target = item["target"]
    print(f"Fetching history for {target}...")
    history_out = run_cmd([
        "/google/bin/releases/flakes-cli/flakes-cli",
        "history",
        target,
        "--days",
        "14",
    ])
    if not history_out:
      continue

    try:
      history = json.loads(history_out)
    except json.JSONDecodeError as e:
      print(f"Failed to decode history JSON for {target}: {e}", file=sys.stderr)
      continue

    has_history_14_days = False
    scores_14 = []
    scores_7 = []
    configs_data = []

    for config_group in history:
      config = config_group.get("configuration", {})
      records = config_group.get("records", [])

      config_str = f"type: {config.get('type')}"
      if "build_flags" in config and config["build_flags"]:
        config_str += f", flags: {', '.join(config['build_flags'])}"

      config_scores_14 = []
      config_scores_7 = []
      for record in records:
        timestamp = record.get("timestamp")
        if not timestamp:
          continue
        try:
          rec_time = datetime.fromisoformat(timestamp)
        except ValueError:
          continue

        score = record.get("score", 0.0)

        if rec_time >= cutoff_14_days:
          config_scores_14.append(score)
          scores_14.append(score)
        if rec_time >= cutoff_7_days:
          config_scores_7.append(score)
          scores_7.append(score)

      if config_scores_14:
        has_history_14_days = True
        config_avg_14 = sum(config_scores_14) / len(config_scores_14)
        config_avg_7 = (
            sum(config_scores_7) / len(config_scores_7)
            if config_scores_7
            else 0.0
        )
        configs_data.append({
            "config": config_str,
            "avg_14": config_avg_14,
            "avg_7": config_avg_7,
        })

    if has_history_14_days:
      avg_14 = sum(scores_14) / len(scores_14) if scores_14 else 0.0
      avg_7 = sum(scores_7) / len(scores_7) if scores_7 else 0.0

      target_data = {
          "target": target,
          "avg_14": avg_14,
          "avg_7": avg_7,
          "configs": configs_data,
          "drill_down": [],
          "links": [],
      }

      if avg_14 > threshold:
        drill_results, links_results = drill_down_target(target)
        target_data["drill_down"] = drill_results
        target_data["links"] = links_results

      report_data.append(target_data)

  standalone_test_cases = []

  for item in report_data:
    simplified_drill_down = []
    shard_failures_count = 0

    for name, count in item["drill_down"]:
      if name.startswith("[test_shard"):
        shard_failures_count += count
      else:
        simplified_drill_down.append((name, count))
        standalone_test_cases.append(
            {"test_case": name, "target": item["target"], "failures": count}
        )

    if shard_failures_count > 0:
      simplified_drill_down.append(
          (f"Shard timeouts/failures", shard_failures_count)
      )

    item["drill_down_simplified"] = simplified_drill_down

  standalone_test_cases.sort(key=lambda x: x["failures"], reverse=True)

  markdown = []
  markdown.append("# Flaky Tests Report for `third_party.py.torch_tpu`")
  markdown.append(f"Generated on: {datetime.now().isoformat()}\n")
  markdown.append(
      "This report lists flaky tests and highlights/drills down into those"
      f" with a 14-day average score above {threshold:.2%}.\n"
  )

  markdown.append("## Summary")
  total_flaky = len(report_data)
  above_threshold = sum(1 for x in report_data if x["avg_14"] > threshold)
  markdown.append(f"- Total flaky targets in last 14 days: {total_flaky}")
  markdown.append(
      f"- Targets with 14-day average flakiness score > {threshold:.2%}:"
      f" {above_threshold}\n"
  )

  markdown.append(f"## Targets with Average Flakiness Score > {threshold:.2%}")
  markdown.append(
      "| Target | Avg Score (14d) | Avg Score (7d) | Details / Shards |"
  )
  markdown.append("| :--- | :---: | :---: | :--- |")

  high_flaky = [x for x in report_data if x["avg_14"] > threshold]
  high_flaky.sort(key=lambda x: x["avg_14"], reverse=True)

  for item in high_flaky:
    details_parts = []
    configs_str = (
        "<b>Configurations (Averages 14d / 7d):</b><br>"
        + "<br>".join([
            f"- {c['config']}: {c['avg_14']:.4%} / {c['avg_7']:.4%}"
            for c in item["configs"]
        ])
    )
    details_parts.append(configs_str)

    shard_failures = [
        f"- `{name}` (failed {count} times)"
        for name, count in item["drill_down_simplified"]
        if "Shard" in name or "Action" in name
    ]

    if shard_failures:
      drill_str = "<b>Recent Execution/Shard Failures:</b><br>" + "<br>".join(
          shard_failures
      )
      details_parts.append(drill_str)
    elif not item["drill_down"]:
      details_parts.append(
          "<i>No recent FAILED executions available for drill-down.</i>"
      )
    else:
      details_parts.append(
          "<i>No shard-level failures (see standalone section for test"
          " cases).</i>"
      )

    details_all = "<br><br>".join(details_parts)
    markdown.append(
        f"| `{item['target']}` | {item['avg_14']:.4%} | {item['avg_7']:.4%} |"
        f" {details_all} |"
    )

  markdown.append("\n## Standalone Flaky Test Cases (Last 14 Days)")
  markdown.append(
      "This section lists specific test methods identified as flaky in recent"
      " runs."
  )
  markdown.append("\n| Test Case | Parent Target | Failures (Recent Runs) |")
  markdown.append("| :--- | :--- | :---: |")
  for tc in standalone_test_cases:
    markdown.append(
        f"| `{tc['test_case']}` | `{tc['target']}` | {tc['failures']} |"
    )

  # Recent Failure Links Section
  markdown.append("\n## Recent Failure Links (Sponge)")
  markdown.append(
      "Below are links to recent failed executions in Sponge for diagnostics"
      " (up to 3 examples per target)."
  )

  has_any_links = False
  for item in report_data:
    if item["links"]:
      has_any_links = True
      markdown.append(f"\n### `{item['target']}`")
      for link in item["links"]:
        markdown.append(
            f"- [Sponge Invocation {link['id']}](http://sponge/{link['id']})"
            f" ({link['status']})"
        )

  if not has_any_links:
    markdown.append("\n*No recent failures available to link.*")

  # Categorization Section
  markdown.append("\n## Issue Categorization")
  markdown.append(
      "Based on the failure signatures, the flaky targets and test cases have"
      " been categorized below:"
  )

  categories = categorize_issues(report_data, standalone_test_cases)

  for cat_name, cat_info in categories.items():
    if cat_info["targets"]:
      markdown.append(f"\n### {cat_name}")
      markdown.append(cat_info["desc"])
      markdown.append("\n**Affected Targets:**")
      for t in cat_info["targets"]:
        t_data = next(x for x in report_data if x["target"] == t)
        markdown.append(f"- `{t}` (Avg 14d: {t_data['avg_14']:.4%})")

      if "test_cases" in cat_info and cat_info["test_cases"]:
        markdown.append("\n**Specific Flaky Test Cases:**")
        for tc in cat_info["test_cases"]:
          markdown.append(
              f"- `{tc['test_case']}` (in `{tc['target']}` - failed"
              f" {tc['failures']} times)"
          )

  markdown.append("\n## All Flaky Targets (Last 14 Days)")
  markdown.append("| Target | Avg Score (14d) | Avg Score (7d) |")
  markdown.append("| :--- | :---: | :---: |")

  report_data.sort(key=lambda x: x["target"])
  for item in report_data:
    target_name = (
        f"**`{item['target']}`**"
        if item["avg_14"] > threshold
        else f"`{item['target']}`"
    )
    markdown.append(
        f"| {target_name} | {item['avg_14']:.4%} | {item['avg_7']:.4%} |"
    )

  if out_path:
    with open(out_path, "w") as f:
      f.write("\n".join(markdown))
    print(f"Report written to {out_path}")
  else:
    print("\n".join(markdown))


if __name__ == "__main__":
  main()
