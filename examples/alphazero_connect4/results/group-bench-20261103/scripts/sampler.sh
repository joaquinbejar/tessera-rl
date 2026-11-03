# Host side: every 60 s, the CPU of every container that is not ours.
set -u
while :; do
  t=$(date +%s)
  docker stats --no-stream --format '{{.Name}} {{.CPUPerc}}' | grep -v '^tessera-m3-' | sed "s/^/$t /"
  sleep 60
done
