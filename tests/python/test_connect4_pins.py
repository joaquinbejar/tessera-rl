# SPDX-License-Identifier: MIT OR Apache-2.0
"""ADR-0042 pins the P2-G1 bot, solver, openings and calibration by SHA-256
(its Evidence table). Any change to a pinned file makes a new bot or a new
protocol, which needs a new ADR: this test fails first."""

import hashlib
import pathlib

import pytest

ROOT = pathlib.Path(__file__).parents[2]

PINS = {
    "tools/connect4-ref/src/bot.rs": "b21c14c26fc77054951c7aa44c139237e0299dc6aa376fa417ae1765bc9ef646",
    "tools/connect4-ref/src/solver.rs": "2927b3f6e49c8defcd5b2348989131cfe2733033cefb44fdce121f2de1e9d19c",
    "tools/connect4-ref/src/board.rs": "664eb16cbbc6d44ddd58a99cfad0e14af6e7dec6c21976f44a255a28bd7a7040",
    "tools/connect4-ref/src/openings.rs": "6cdd2ae7dd9d6fbf6b49c54aace88811b214a21e1acf7620fb2deb4d3a38c218",
    "tools/connect4-ref/src/rng.rs": "907f5f89679dab32cd8a6f6e982b59984e4113dae92a7946aaf0287a0c1255b2",
    "tools/connect4-ref/src/calibrate.rs": "4d97d0cdeb8c013747c81d6ad67b178cfe0d3fb51e924cc2f0755e98697e08ca",
    "compat/connect4/openings-eval.txt": "54a1cea7e6fe70d53a039bc27176486953c8bb6a243353175128526702360475",
    "compat/connect4/openings-heldout.txt": "3276d2d044c911ed29c36f2d42690697f19a4bac0183e3a77b1af3c5af72cc49",
}


@pytest.mark.parametrize("path", sorted(PINS))
def test_connect4_pins_adr0042_files_are_byte_identical(path):
    digest = hashlib.sha256((ROOT / path).read_bytes()).hexdigest()
    assert digest == PINS[path], f"{path} changed: a pinned ADR-0042 file needs a new ADR"
