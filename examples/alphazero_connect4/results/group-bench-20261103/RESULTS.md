<!-- SPDX-License-Identifier: MIT OR Apache-2.0 -->
# ADR-0045 section 5 measurement, 2026-11-03

This is the one-shot, pre-registered SearchGroup measurement for self-play ([GROUP_BENCH.md](../../GROUP_BENCH.md), ADR-0045 section 5 and its addendum 1). It ran on swarmer02 under the coordinator's go, after both reviewers approved, and it was not rerun.

## Verdict

The verdict block of `group-bench.json`, verbatim:

```json
{"adopted": true, "criterion": "group T=8 W=32 median >= 1.2x single-tree W=32 median, games per hour per allocated core", "diagnostic_ratio_w1": 2.077275483495703, "formal": true, "normalization": "every cell, W = 1 included, divides by the 32 allocated cores; the W = 1 ratio is unaffected, its absolute rates are not per busy core", "not_formal_because": [], "ratio": 1.8264066168803306}
```

The ratio is **observed shared-host throughput**: the host's production services ran alongside, as its normal state (see "Other containers"). It is not a speedup of the isolated engine.

swarmer02 has 16 physical cores and 32 SMT threads, and the 32 "allocated cores" are logical CPUs. The 1.83x is consistent with SMT and shared-resource contention, since single-tree games per busy second fall from 5.14 at W 8 to 2.04 at W 32. That share is not quantified; it is not a measured causal part of the ratio. Do not extrapolate the ratio to hosts without SMT or to other core counts.

SearchGroup is adopted but not used in the PR(3) run. The measurement covers the small net, the medium net was not starved, and the single-tree path that PR(2) validated is kept. SearchGroup stays in the codebase.

## Cells

Each cell kept 7 rounds after 2 warmup rounds. Q = 32 logical games per worker per round, and no cell stopped.

| Cell | Median games/h per allocated core | Median games per busy wall second (companion) |
| --- | --- | --- |
| single, W 1 | 636.6 | 5.66 |
| single, W 8 | 4,177.2 | 5.14 |
| single, W 32 | 6,870.4 | 2.04 |
| group, W 32, T 4 | 10,760.8 | 3.14 |
| group, W 32, T 8 | 12,548.1 | 3.68 |
| group, W 32, T 16 | 9,728.3 | 2.88 |
| group, W 1, T 8 | 1,322.3 | 11.76 |

The matched-work companion uses 64 fixed roots, 100 simulations, both modes warm, and 7 alternating kept rounds each. Its median pass took 1.249 s single and 0.794 s group, a speedup of 1.572, and it did not stop. Companion figures are evidence only; they are never the criterion.

## Checkpoint: platform-specific initialization provenance

The small net is `train_step.py init --size small --seed 20261102`.
- In the pinned Linux image on swarmer02 its digest is `80998924a6423e7ed515d64e1be849a439dbb9d53b8555618534b1397e5ed5c6` (`S.sha256`, `S.json`, `init.log`). That is the same checkpoint the P2-M3 pilot built on Linux.
- On macOS arm64 the same command gives `c0fc7e29…ec3e`.

The initialization is therefore platform-specific. It was observed in a separate init-only run, and the Linux digest was registered in addendum 1 before any measurement; every cell used that one file. This records the provenance of the initialization. It is not a regression in environment determinism.

## Other containers

`scripts/sampler.sh` sampled `docker stats --no-stream` of every non-tessera container every 60 s during the run. The samples are in `others-cpu.log`, raw, with their epochs. The other services belong to the host owner, so the committed copy names each container only by the first 8 hex digits of the SHA-256 of its name; the CPU values and epochs are unchanged.
- Samples: 10.
- Summed CPU of the others: maximum 2.83 %, mean 1.62 %. In docker's units 100 % is one CPU, so this is never more than about 0.03 of one CPU.
- No sample exceeded 100 %, so the verdict carries no caveat.

The run started at epoch 1793705854 (`launch-measure.json`), and the cells ran in their declared order. The ledger phase started at 11:37:41Z and lasted 606.4 s.

## Run and budget

| Item | Value |
| --- | --- |
| Ledger entry | `group-bench 20261103T113741Z`, ok, 606.4 s wall, 5.390 of 24 core-hours |
| Container | `tessera-m3-gb-measure`, docker exit 0, 614 s wall from setup to exit, cap 3,300 s |
| Init-only container | `tessera-m3-gb-init`, docker exit 0, 34 s |
| Runtime (self-declared) | host swarmer02, image `rust:1.99.1@sha256:09027f87…0137`, cpuset 0-31, affinity 32, torch 1/1 threads, deterministic, boot_id `70d3e9d8-9211-459c-b071-6eebd8d70c6f`, AMD Ryzen 9 9955HX 16-Core, MemTotal 94,360,388 kB |

Before the launch, `pre.sh` reported:
- RepoDigests `["rust@sha256:09027f873d45b8ecd1dc86f1fa13b7a62b875c961f017a0e331267b350430137"]`;
- load average 0.63, 0.67, 0.68;
- the production containers running, and no tessera container;
- 17 named volumes, none of them ours.

The full `pre.sh` listings name the owner's private services, so they are not committed; the coordinator has them on the bus. The host name and image are self-declared.

Cleanup:
- The evidence was copied to the Mac first.
- Then `~/tessera-rl-m3gb` was removed and the image deleted with `docker rmi`.
- `docker ps -a --filter name=tessera-m3` was empty, and the volume set was unchanged (17). The trap in `run.sh` reported neither a leftover container nor a changed volume set.

## Wrapper changes after registration

`scripts/run.sh` went through three versions. The one that ran the measurement was frozen and announced (ACS #tessera-rl, msg 50587) before `sh run.sh measure` started.

1. `8fa55c49…`: the first draft (msg 50450). It never ran.
2. `80d63cda8f4164b2e8d490cb026658a92ea9e620d77e901033b04f1c911c255d`: the coordinator's fixes (msg 50455), which are absolute bind-mount paths, `export S1` and the volume-set check. This version ran the init-only step (`launch-init.json`) and is the one registered in msg 50479.
3. `26447f8cb5bb28bc4debce140108965e0494879096091edb9720a184f90e4589`: the measurement wrapper. It adds only the host-side CPU sampler that the coordinator's go required (msg 50488). The whole diff from 80d63cda:

```diff
@@ -11,7 +11,9 @@
 cd "$DIR" || exit 2
 export S1
 VOLUMES_BEFORE=$(docker volume ls -q | sort)
+SAMPLER=
 cleanup() {
+  [ -n "$SAMPLER" ] && kill "$SAMPLER" 2>/dev/null
   docker rm -f "$NAME" >/dev/null 2>&1
@@ -22,6 +24,10 @@
 }
 trap cleanup EXIT INT TERM HUP
 mkdir -p out
+if [ "$PHASE" = measure ]; then
+  sh "$DIR/sampler.sh" >> "$DIR/out/others-cpu.log" 2>&1 &
+  SAMPLER=$!
+fi
 start=$(date +%s)
```

The diff leaves these unchanged: the `docker run` line, mounts, environment, cap, timeout and trap logic, `inner-measure.sh` (`38e9cdc5…`), `pre.sh` (`c320dce1…`) and the measurement protocol. `sampler.sh` is `ea33fbb8faeeadf80d2b24bc60e901999107b973c557ded74f5f189a40bd7115`. Every script was checked with `sha256sum` on the host after upload, and the committed `scripts/` are those exact files.

## Addendum 1 records

| Record | Value |
| --- | --- |
| Source S1 | `46462481eb1fc659022d0fe7d56f3c86b0ef9cb8` |
| `git archive --format=tar` sha256 | `2dabd40a2c830fc66c703d62459b5e0aea00c633d8c7a43f602ec2ff2f74e74b`, verified on the host |
| Harness diff from a835985 | empty |
| config_sha256 (pre-registered, equal to the report's) | `130e07d01fa31e1d004774e1ccb3c3e12a9c13dbbb8561b4a22891ad75a19ea2` |
| Checkpoint (registered Linux init) | `80998924a6423e7ed515d64e1be849a439dbb9d53b8555618534b1397e5ed5c6` |
| `group-bench.json` sha256 (uncompressed; shipped as `group-bench.json.gz`, gzip -9 -n) | `1762559b281c5eed75841405c55308f4f71029cfb2dca3d24c173b11e02869a5` |
| `ledger.json` sha256 | `e00184cd2cb0a2295f149728e97176ca5c9f9b73ac4fa2619fb57809e24daa50` |
| `others-cpu.log` sha256 | `93b0f2d6f82fc32c7296a8351a49333fef7c58cb1230ebc9817a13733d7dce00` as recorded; the committed, pseudonymized copy is `8c3770b50ade78b45dd81a509f25187dcd16e219aec2ba4260b5d26fb3bec2ee` |
| `scripts/run.sh` sha256 | `26447f8cb5bb28bc4debce140108965e0494879096091edb9720a184f90e4589` |
| `scripts/sampler.sh` sha256 | `ea33fbb8faeeadf80d2b24bc60e901999107b973c557ded74f5f189a40bd7115` |
| `scripts/pre.sh` sha256 | `c320dce110326b7f18fd64d05ac99b758ab92cb78fa8b95f6f4f59cf515164b8` |
| `scripts/inner-init.sh` sha256 | `032aca805b08553e3547d340c86de5cdffa7553c134b298b98d6d41ea9c14830` |
| `scripts/inner-measure.sh` sha256 | `38e9cdc529948355e302a1668344410796f0b342e80cd52d8fae668239de96c7` |
| uv binary (x86-64, from xaskar) sha256 | `23803452c0fbe125033054cd9ff245f1d649fb65b4b73f68429f4abda432436d` |
