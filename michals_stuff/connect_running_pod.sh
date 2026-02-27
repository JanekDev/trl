#!/usr/bin/env bash
set -euo pipefail

if ! command -v runpodctl >/dev/null 2>&1; then
  echo "runpodctl is not installed or not in PATH."
  exit 1
fi

extract_pod_ids() {
  if command -v jq >/dev/null 2>&1; then
    jq -r '
      def rows:
        if type == "array" then .
        elif type == "object" then (.pods // .items // .data // .results // [])
        else [] end;
      rows[]
      | .id?
      | strings
      | select(length > 0)
    '
    return
  fi

  if command -v python3 >/dev/null 2>&1; then
    python3 - <<'PY'
import json, sys

try:
    data = json.load(sys.stdin)
except Exception:
    sys.exit(1)

def rows(obj):
    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict):
        out = []
        for k in ("pods", "items", "data", "results"):
            v = obj.get(k)
            if isinstance(v, list):
                out.extend(v)
        return out
    return []

for row in rows(data):
    if isinstance(row, dict):
        pod_id = row.get("id")
        if isinstance(pod_id, str) and pod_id:
            print(pod_id)
PY
    return
  fi

  echo "Need either jq or python3 to parse runpodctl JSON output." >&2
  return 1
}

running_ids=()
while IFS= read -r pod_id; do
  [[ -n "$pod_id" ]] && running_ids+=("$pod_id")
done < <(runpodctl pod list --status RUNNING -o json | extract_pod_ids)

if [[ "${#running_ids[@]}" -eq 0 ]]; then
  echo "No running pods found."
  exit 1
fi

if [[ "${#running_ids[@]}" -gt 1 ]]; then
  echo "More than one running pod found. This script expects exactly one:"
  printf '  %s\n' "${running_ids[@]}"
  echo "Stop/delete extras or connect manually with: runpodctl ssh info <pod-id>"
  exit 1
fi

pod_id="${running_ids[0]}"
ssh_info="$(runpodctl ssh info "$pod_id" -o json)"

ssh_cmd=""
if command -v jq >/dev/null 2>&1; then
  ssh_cmd="$(printf '%s\n' "$ssh_info" | jq -r '.ssh_command // empty')"
elif command -v python3 >/dev/null 2>&1; then
  ssh_cmd="$(printf '%s\n' "$ssh_info" | python3 - <<'PY'
import json, sys
try:
    data = json.load(sys.stdin)
except Exception:
    print("")
    raise SystemExit(0)
print(data.get("ssh_command", "") if isinstance(data, dict) else "")
PY
)"
fi

# Fallback for non-JSON output formats.
if [[ -z "$ssh_cmd" ]]; then
  ssh_cmd="$(printf '%s\n' "$ssh_info" | sed -n -E 's/^[[:space:]]*(ssh[[:space:]].*)$/\1/p' | head -n1)"
fi

if [[ -z "$ssh_cmd" ]]; then
  echo "Could not detect an ssh command in runpodctl ssh info output."
  echo "Pod ID: $pod_id"
  echo
  printf '%s\n' "$ssh_info"
  exit 1
fi

if [[ "${1:-}" == "--print" ]]; then
  echo "$ssh_cmd"
  exit 0
fi

echo "Connecting to pod: $pod_id"
echo "$ssh_cmd"
exec bash -lc "$ssh_cmd"
