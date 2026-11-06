#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
"""Oracle and disagreement report of a best-response run: `OUT C6_LOOK_LOG`.

Reads only. Oracle: no slot that C6 or the solver won may be exhausted. Any such
slot is a tool bug and voids the run. Disagreement is indicative: lines are not unique.
"""
import hashlib
import json
import sys

out, c6_path = sys.argv[1], sys.argv[2]
results = {}
for row in open(f"{out}/results.tsv"):
    f = row.rstrip("\n").split("\t")
    results[(int(f[0]), f[1])] = f[2]

c6 = {}
for j, first, score, _ in json.load(open(c6_path)):
    c6[(j, "first" if first else "second")] = score
c6_wins = [k for k, v in c6.items() if v == 1.0]
solver_wins = []
for row in open(f"{out}/solver.tsv"):
    if row.startswith("#"):
        continue
    f = row.split()
    if f[2] == "win":
        solver_wins.append((int(f[0]), f[1]))
bad_c6 = [k for k in c6_wins if results.get(k) == "exhausted"]
bad_solver = [k for k in solver_wins if results.get(k) == "exhausted"]
c6_missed_w = [k for k in c6_wins if results.get(k) != "win"]

HELDOUT = "compat/connect4/openings-heldout.txt"
HELDOUT_SHA256 = "3276d2d044c911ed29c36f2d42690697f19a4bac0183e3a77b1af3c5af72cc49"
if hashlib.sha256(open(HELDOUT, "rb").read()).hexdigest() != HELDOUT_SHA256:
    sys.exit("the held-out file is not the pinned one")
openings = {}
for row in open(HELDOUT):
    if not row.startswith("#"):
        f = row.split()
        openings[int(f[0])] = f[2]
# Keyed by the board, not the move order: different openings have different tie
# seeds, so only transpositions can meet. Colours are by ply parity.
seen = {}
for row in open(f"{out}/lines.txt"):
    j, colour, _, moves = row.split()
    j = int(j)
    board = [""] * 7
    for ply, c in enumerate(openings[j]):
        board[int(c) - 1] += str(ply % 2)
    ply = len(openings[j])
    cols = moves.split(",")
    for i, c in enumerate(cols):
        candidate = (i % 2 == 0) == (colour == "first")
        if candidate:
            seen.setdefault(tuple(board), {}).setdefault(c, set()).add((j, colour))
        board[int(c)] += str(ply % 2)
        ply += 1
conflicts = sum(1 for v in seen.values() if len(v) > 1)
# How many comparisons the figure rests on: a position one slot reached compares nothing.
shared = sum(1 for v in seen.values() if len(set().union(*v.values())) > 1)

print(json.dumps({
    "c6_slots": len(c6), "c6_wins": len(c6_wins),
    "solver_wins": len(solver_wins),
    "oracle_c6_violations": bad_c6, "oracle_solver_violations": bad_solver,
    "c6_won_not_win_class": len(c6_missed_w),
    "candidate_positions": len(seen), "positions_in_several_slots": shared,
    "positions_with_disagreeing_moves": conflicts,
}))
