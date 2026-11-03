#!/bin/sh
# Host side, on swarmer02 in ~/tessera-rl-m3gb: sh run.sh init|measure
set -u
PHASE=$1
D=09027f873d45b8ecd1dc86f1fa13b7a62b875c961f017a0e331267b350430137
IMG=rust:1.99.1@sha256:$D
S1=46462481eb1fc659022d0fe7d56f3c86b0ef9cb8
NAME=tessera-m3-gb-$PHASE
case $PHASE in init) CAP=900 ;; measure) CAP=3300 ;; *) echo "usage: run.sh init|measure" >&2; exit 2 ;; esac
DIR=$(cd "$(dirname "$0")" && pwd) || exit 2
cd "$DIR" || exit 2
export S1
VOLUMES_BEFORE=$(docker volume ls -q | sort)
SAMPLER=
cleanup() {
  [ -n "$SAMPLER" ] && kill "$SAMPLER" 2>/dev/null
  docker rm -f "$NAME" >/dev/null 2>&1
  if docker ps -a --filter "name=^${NAME}\$" --format '{{.Names}}' | grep -q .; then
    echo "CONTAINER LEFT: $NAME" >&2
  fi
  if [ "$(docker volume ls -q | sort)" != "$VOLUMES_BEFORE" ]; then
    echo "VOLUME SET CHANGED" >&2
  fi
}
trap cleanup EXIT INT TERM HUP
mkdir -p out
if [ "$PHASE" = measure ]; then
  sh "$DIR/sampler.sh" >> "$DIR/out/others-cpu.log" 2>&1 &
  SAMPLER=$!
fi
start=$(date +%s)
timeout --kill-after 30 "$CAP" docker run --rm --name "$NAME" --hostname swarmer02 --cpuset-cpus 0-31 \
  -e PILOT_IMAGE_DIGEST="$IMG" -e PILOT_CPUSET=0-31 -e S1 -e CKPT_SHA="${CKPT_SHA:-}" \
  -e HOST_UID="$(id -u)" -e HOST_GID="$(id -g)" \
  -v "$DIR/src:/work/tessera" -v "$DIR/out:/out" -v "$DIR/uv:/usr/local/bin/uv:ro" \
  -v "$DIR/inner-$PHASE.sh:/inner.sh:ro" -w /work/tessera \
  -e UV_CACHE_DIR=/work/tessera/.uvcache -e UV_LINK_MODE=copy \
  "$IMG" sh /inner.sh > "$DIR/out/$PHASE.log" 2>&1
rc=$?
end=$(date +%s)
printf '{"phase":"%s","docker_exit":%d,"container_wall_s":%d,"started_epoch":%d,"cap_s":%d}\n' \
  "$PHASE" "$rc" $((end - start)) "$start" "$CAP" > "$DIR/out/launch-$PHASE.json"
exit "$rc"
