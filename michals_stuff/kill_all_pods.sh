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

yes_flag=false
if [[ "${1:-}" == "--yes" || "${1:-}" == "-y" ]]; then
  yes_flag=true
fi

pod_ids=()
while IFS= read -r pod_id; do
  [[ -n "$pod_id" ]] && pod_ids+=("$pod_id")
done < <(runpodctl pod list -a -o json | extract_pod_ids)

if [[ "${#pod_ids[@]}" -eq 0 ]]; then
  echo "No pods found."
  exit 0
fi

echo "Found ${#pod_ids[@]} pod(s):"
printf '  %s\n' "${pod_ids[@]}"

if [[ "$yes_flag" != true ]]; then
  read -r -p "Delete all of these pods? [y/N] " reply
  case "$reply" in
    y|Y|yes|YES) ;;
    *) echo "Cancelled."; exit 1 ;;
  esac
fi

failures=0
for pod_id in "${pod_ids[@]}"; do
  echo "Deleting pod: $pod_id"
  if ! runpodctl pod delete "$pod_id"; then
    failures=$((failures + 1))
    echo "Failed to delete pod: $pod_id" >&2
  fi
done

if [[ "$failures" -gt 0 ]]; then
  echo "Finished with $failures failure(s)." >&2
  exit 1
fi

echo "All pods deleted."
