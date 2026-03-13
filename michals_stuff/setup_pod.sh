#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if ! command -v runpodctl >/dev/null 2>&1; then
  echo "runpodctl is not installed or not in PATH."
  echo "Install it first, then re-run this script."
  exit 1
fi

POD_NAME="${POD_NAME:-mtpi-$(date +%Y%m%d-%H%M%S)}"
POD_PROFILE="${POD_PROFILE:-h100-sxm-secure}" # h100-sxm-secure | h100-nvl-community | a100-community | md | lg | xl
GPU_COUNT="${GPU_COUNT:-1}"
IMAGE_NAME="${IMAGE_NAME:-runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04}"
CONTAINER_DISK_SIZE="${CONTAINER_DISK_SIZE:-100}"
VOLUME_SIZE="${VOLUME_SIZE:-0}"
VOLUME_PATH="${VOLUME_PATH:-/workspace}"
PORTS="${PORTS:-22/tcp,8888/http}"
RUNPOD_CLOUD="${RUNPOD_CLOUD:-secure}" # community | secure
PUBLIC_IP="${PUBLIC_IP:-true}"            # true | false (community only)
DATA_CENTER_IDS="${DATA_CENTER_IDS:-}"    # comma-separated ids
ENV_JSON="${ENV_JSON:-}"                  # JSON object string, e.g. '{"WANDB_API_KEY":"..."}'
DOTENV_PATH="${DOTENV_PATH:-$SCRIPT_DIR/.env}"
GPU_ID="${GPU_ID:-}"
GPU_TYPE="${GPU_TYPE:-}"                  # backwards compatibility alias
COST_CEILING="${COST_CEILING:-}"          # unsupported by current runpodctl pod create

if [[ -z "$GPU_ID" && -n "$GPU_TYPE" ]]; then
  GPU_ID="$GPU_TYPE"
fi

if [[ -z "$GPU_ID" ]]; then
  case "$POD_PROFILE" in
    h100-sxm-secure)
      GPU_ID="NVIDIA H100 80GB HBM3"
      ;;
    h100-nvl-community)
      GPU_ID="NVIDIA H100 NVL"
      ;;
    a100-community)
      GPU_ID="NVIDIA A100-SXM4-80GB"
      ;;
    md)
      GPU_ID="NVIDIA L40S"
      ;;
    lg)
      GPU_ID="NVIDIA A100 80GB PCIe"
      ;;
    xl)
      GPU_ID="NVIDIA H100 80GB HBM3"
      ;;
    *)
      echo "Unsupported POD_PROFILE='$POD_PROFILE'. Use: h100-sxm-secure | h100-nvl-community | a100-community | md | lg | xl"
      exit 1
      ;;
  esac
fi

RUNPOD_CLOUD_LC="$(printf '%s' "$RUNPOD_CLOUD" | tr '[:upper:]' '[:lower:]')"
PUBLIC_IP_LC="$(printf '%s' "$PUBLIC_IP" | tr '[:upper:]' '[:lower:]')"

case "$RUNPOD_CLOUD_LC" in
  community)
    CLOUD_TYPE="COMMUNITY"
    ;;
  secure)
    CLOUD_TYPE="SECURE"
    ;;
  *)
    echo "RUNPOD_CLOUD must be 'community' or 'secure' (got '$RUNPOD_CLOUD')."
    exit 1
    ;;
esac

MERGED_ENV_JSON=""
if [[ -f "$DOTENV_PATH" || -n "$ENV_JSON" ]]; then
  if ! command -v python3 >/dev/null 2>&1; then
    echo "python3 is required to parse $DOTENV_PATH for --env." >&2
    exit 1
  fi
  MERGED_ENV_JSON="$(
    python3 - "$DOTENV_PATH" "$ENV_JSON" <<'PY'
import json
import os
import sys

dotenv_path = sys.argv[1]
env_json = sys.argv[2]
merged = {}

if os.path.isfile(dotenv_path):
    with open(dotenv_path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[len("export "):].strip()
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip()
            if not key:
                continue
            if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
                value = value[1:-1]
            merged[key] = value

if env_json:
    try:
        extra = json.loads(env_json)
    except json.JSONDecodeError as e:
        print(f"Invalid ENV_JSON: {e}", file=sys.stderr)
        sys.exit(2)
    if not isinstance(extra, dict):
        print("ENV_JSON must be a JSON object.", file=sys.stderr)
        sys.exit(2)
    for k, v in extra.items():
        merged[str(k)] = "" if v is None else str(v)

print(json.dumps(merged, separators=(",", ":")))
PY
  )"
fi

cmd=(
  runpodctl pod create
  --name "$POD_NAME"
  --cloud-type "$CLOUD_TYPE"
  --gpu-id "$GPU_ID"
  --gpu-count "$GPU_COUNT"
  --image "$IMAGE_NAME"
  --container-disk-in-gb "$CONTAINER_DISK_SIZE"
  --ports "$PORTS"
  --ssh
)

if [[ "$VOLUME_SIZE" != "0" ]]; then
  cmd+=(--volume-in-gb "$VOLUME_SIZE" --volume-mount-path "$VOLUME_PATH")
fi

if [[ "$CLOUD_TYPE" == "COMMUNITY" && "$PUBLIC_IP_LC" == "true" ]]; then
  cmd+=(--public-ip)
fi

if [[ -n "$DATA_CENTER_IDS" ]]; then
  cmd+=(--data-center-ids "$DATA_CENTER_IDS")
fi

if [[ -n "$MERGED_ENV_JSON" && "$MERGED_ENV_JSON" != "{}" ]]; then
  cmd+=(--env "$MERGED_ENV_JSON")
fi

echo "Creating pod with:"
echo "  name=$POD_NAME"
echo "  profile=$POD_PROFILE"
echo "  cloud=$RUNPOD_CLOUD"
echo "  gpu=$GPU_ID x$GPU_COUNT"
echo "  image=$IMAGE_NAME"
if [[ "$VOLUME_SIZE" == "0" ]]; then
  echo "  disk=${CONTAINER_DISK_SIZE}GB container, no persistent volume"
else
  echo "  disk=${CONTAINER_DISK_SIZE}GB container, ${VOLUME_SIZE}GB volume at $VOLUME_PATH"
fi
echo "  ports=$PORTS"
echo "  public_ip=$PUBLIC_IP"
if [[ -n "$DATA_CENTER_IDS" ]]; then
  echo "  data_centers=$DATA_CENTER_IDS"
fi
if [[ -n "$MERGED_ENV_JSON" && "$MERGED_ENV_JSON" != "{}" ]]; then
  if [[ -f "$DOTENV_PATH" && -n "$ENV_JSON" ]]; then
    echo "  env=merged from $DOTENV_PATH + ENV_JSON (ENV_JSON overrides)"
  elif [[ -f "$DOTENV_PATH" ]]; then
    echo "  env=from $DOTENV_PATH"
  else
    echo "  env=from ENV_JSON"
  fi
fi
if [[ -n "$COST_CEILING" ]]; then
  echo "  note: COST_CEILING is ignored by current runpodctl pod create"
fi

echo
"${cmd[@]}"

echo
echo "Pod requested."
echo "Next:"
echo "1) Check status: runpodctl pod list"
echo "2) Once RUNNING, connect using Runpod SSH info (or runpodctl ssh)."
echo "3) Inside pod, run your setup and experiments."
