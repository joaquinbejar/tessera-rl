# Container side: the small checkpoint only, no measurement.
set -eu
trap 'chown -R "$HOST_UID:$HOST_GID" /work/tessera /out' EXIT
R="uv run --locked --python 3.11 --group test --group torch"
uv sync --locked --python 3.11 --group test --group torch
$R python examples/alphazero_connect4/train_step.py init --size small --seed 20261102 --out /out/S.pt --source-sha "$S1"
$R python -c 'import sys; sys.path.insert(0, "examples/alphazero_connect4"); import ckpt; print(ckpt.manifest("/out/S.pt")["sha256"])' > /out/S.sha256
cat /out/S.sha256
