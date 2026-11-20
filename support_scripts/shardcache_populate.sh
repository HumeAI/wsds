#!/usr/bin/env bash
# Fill the node-local shard cache from the shared-filesystem mirrors: one SLURM job per ring node,
# each copying the mirrors that node owns onto its own drives (see wsds/shardcache.py).
#
#   python -m wsds.shardcache scan /mnt/weka/data-wsds > mirrors.txt     # once; any list of *.sparse paths works
#   support_scripts/shardcache_populate.sh ring.json mirrors.txt         # copy
#   support_scripts/shardcache_populate.sh ring.json mirrors.txt --consume   # copy, then delete merged sources
#
# Safe to re-run: a second pass only copies blocks the node copy does not have yet.
# The ring file, the list and this checkout must be on a filesystem every node sees.
# Environment: PYTHON (interpreter with wsds's dependencies), PARTITION (default gpu), LOGDIR.
set -euo pipefail

if [ $# -lt 2 ]; then
    sed -n '2,11p' "$0" >&2
    exit 2
fi
ring=$(realpath "$1")
list=$(realpath "$2")
shift 2
repo=$(cd "$(dirname "$0")/.." && pwd)
py=${PYTHON:-$(command -v python3)}
logdir=${LOGDIR:-$PWD/shardcache-populate-logs}
mkdir -p "$logdir"

hosts=$("$py" -c 'import json, sys; print(" ".join(n["host"] for n in json.load(open(sys.argv[1]))["nodes"]))' "$ring")
for host in $hosts; do
    # One GPU per job: on this cluster the GPU is the unit of allocation, even for an IO job.
    sbatch -p "${PARTITION:-gpu}" -G 1 -w "$host" -J "shardcache-$host" -o "$logdir/$host.log" \
        --wrap "cd '$repo' && '$py' -m wsds.shardcache --ring '$ring' populate --list '$list' $*"
done
echo "logs: $logdir/<host>.log"
