# scripts/insert_connectivity_check.py
# Idempotent script to insert a composite action step into all workflows under .github/workflows
# Usage: python3 scripts/insert_connectivity_check.py .github/workflows

import sys, os, yaml, shutil

CONNECTIVITY_STEP = {
  "name": "Check GitHub Actions connectivity",
  "uses": "./.github/actions/check-actions-connectivity"
}

def process_workflow(path):
  with open(path, "r", encoding="utf-8") as f:
    data = yaml.safe_load(f)

  if not data or "jobs" not in data:
    return False

  changed = False
  for job_name, job in data["jobs"].items():
    steps = job.get("steps")
    if not isinstance(steps, list):
      continue
    # detect existing connectivity step
    if any((isinstance(step, dict) and (step.get("uses") == CONNECTIVITY_STEP["uses"] or step.get("name") == CONNECTIVITY_STEP["name"])) for step in steps):
      continue
    # find index of first uses: that references a remote action (heuristic) or insert at pos 0
    insert_at = 0
    for i, step in enumerate(steps):
      if isinstance(step, dict) and "uses" in step:
        insert_at = i
        break
    steps.insert(insert_at, CONNECTIVITY_STEP)
    job["steps"] = steps
    changed = True

  if changed:
    # backup original
    shutil.copy2(path, path + ".bak")
    with open(path, "w", encoding="utf-8") as f:
      yaml.safe_dump(data, f, default_flow_style=False, sort_keys=False)
    print(f"Patched {path}")
  return changed


def main():
  if len(sys.argv) < 2:
    print("Usage: python3 insert_connectivity_check.py .github/workflows")
    sys.exit(1)
  root = sys.argv[1]
  for rootdir, _, files in os.walk(root):
    for fn in files:
      if fn.endswith((".yml", ".yaml")):
        full = os.path.join(rootdir, fn)
        try:
          process_workflow(full)
        except Exception as e:
          print(f"Error processing {full}: {e}")

if __name__ == "__main__":
  main()
