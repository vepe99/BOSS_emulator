#!/usr/bin/env bash
# After PID $1 exits, turn checkpoint symlinks (redirected to local disk while NFS was slow)
# in the given run dirs back into real files on NFS.   usage: ckpt_writeback.sh <pid> <run_dir>...
pid=$1; shift
while kill -0 "$pid" 2>/dev/null; do sleep 10; done
for d in "$@"; do
  for f in "$d"/*.weights.h5; do
    [ -L "$f" ] || continue
    t=$(readlink -f "$f")
    if [ -f "$t" ]; then cp "$t" "$f.tmp" && mv -T "$f.tmp" "$f" && echo "[$(date +%T)] wrote back $f"
    else rm "$f" && echo "[$(date +%T)] removed dangling $f"; fi
  done
done
echo "[$(date +%T)] writeback done"
