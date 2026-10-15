#!/usr/bin/env bash
# Running/Pending pods in rationai-jobs-ns as CSV (name, status, gpus, gpu type, node, started).
# Usage: scripts/pods_csv.sh [output.csv]   (no arg = print to stdout)
set -euo pipefail
NS="${KUBE_NAMESPACE:-rationai-jobs-ns}"

kubectl get pods -n "$NS" -o json | python3 -c '
import csv, json, sys
rows = [["pod","status","gpu_count","gpu_product","node","started"]]
for p in json.load(sys.stdin)["items"]:
    if p["status"].get("phase") not in ("Running","Pending"): continue
    g = sum(int(c.get("resources",{}).get("requests",{}).get("nvidia.com/gpu",0)) for c in p["spec"]["containers"])
    rows.append([p["metadata"]["name"], p["status"]["phase"], g,
        p["spec"].get("nodeSelector",{}).get("nvidia.com/gpu.product","") if g else "",
        p["spec"].get("nodeName","unscheduled"), p["status"].get("startTime","")])
rows[1:] = sorted(rows[1:], key=lambda r:(r[1],r[0]))
out = open(sys.argv[1], "w", newline="") if len(sys.argv) > 1 else sys.stdout
csv.writer(out).writerows(rows)
' "${1:--}"
