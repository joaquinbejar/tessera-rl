# Host side, right before each launch: the self-declared evidence for the announcement.
set -u
IMG=rust:1.99.1@sha256:09027f873d45b8ecd1dc86f1fa13b7a62b875c961f017a0e331267b350430137
docker pull -q "$IMG" >/dev/null
echo "RepoDigests: $(docker inspect --format '{{json .RepoDigests}}' "$IMG")"
echo "docker ps -a:"; docker ps -a --format '{{.Names}} {{.Image}} {{.Status}}'
echo "uptime: $(uptime)"
echo "nproc: $(nproc)  hostname: $(hostname)"
echo "docker volume ls -q:"; docker volume ls -q
