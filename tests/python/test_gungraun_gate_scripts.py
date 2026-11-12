# SPDX-License-Identifier: MIT OR Apache-2.0
"""The instruction-count gate's profile and two-sided band (ADR-0059): the band
script on fake gungraun output (colours on and off, each side of the band, a
missing case) and the profile check on fake `cargo -v` output."""

import os
import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
BAND = ROOT / "scripts" / "gungraun_band.sh"
PROFILE = ROOT / "scripts" / "gungraun_profile_check.sh"
GATE = ROOT / "scripts" / "gungraun.sh"

pytestmark = pytest.mark.skipif(os.name == "nt", reason="POSIX only: bash scripts")

BASE = 1_000_000
CASES = ["games_instructions::games::connect4_pool_step n1024", "search_instructions::search::puct_run sims_256",
         "classic_instructions::classic::cartpole_physics steps_1000"]


def _log(new_values, colour):
    """gungraun's check output for CASES, with `new_values` against BASE."""
    bold, green, reset = ("\x1b[1m", "\x1b[32m", "\x1b[0m") if colour else ("", "", "")
    out = []
    for case, new in zip(CASES, new_values):
        out.append(f"{green}{case}{reset}:(setup())")
        out.append(f"  Instructions:                                    {bold}{new}{reset}|{BASE}              (+0.00%)")
        out.append("  L1 Hits:                                          100|100                  (No change)")
    return "\n".join(out) + "\n"


def _band(tmp_path, new_values, colour, expected=len(CASES)):
    log = tmp_path / "gungraun.log"
    log.write_text(_log(new_values, colour))
    return subprocess.run(["bash", str(BAND), str(log), "2.0", str(expected)], capture_output=True, text=True, check=False)


@pytest.mark.parametrize("colour", [False, True], ids=["plain", "colour"])
@pytest.mark.parametrize("delta", [-20_000, -19_999, 0, 19_999, 20_000], ids=["-2.000", "-1.999", "0", "+1.999", "+2.000"])
def test_gungraun_band_inside_or_on_the_band_passes(tmp_path, colour, delta):
    p = _band(tmp_path, [BASE + delta, BASE, BASE], colour)
    assert p.returncode == 0, p.stderr
    assert "band check passed: 3 cases" in p.stdout


@pytest.mark.parametrize("colour", [False, True], ids=["plain", "colour"])
def test_gungraun_band_just_below_fails_and_asks_for_a_rebless(tmp_path, colour):
    p = _band(tmp_path, [BASE - 20_001, BASE, BASE], colour)
    assert p.returncode == 1
    assert "below the baseline by more than 2.0%" in p.stderr
    assert "connect4_pool_step n1024: 979999 against the baseline 1000000" in p.stderr
    assert "puct_run" not in p.stderr
    assert "dispatch the 'gungraun bless' workflow" in p.stderr
    assert "commit the artifact's files unchanged" in p.stderr
    assert "run id and the artifact's sha256" in p.stderr


@pytest.mark.parametrize("colour", [False, True], ids=["plain", "colour"])
def test_gungraun_band_just_above_fails_as_a_regression(tmp_path, colour):
    p = _band(tmp_path, [BASE, BASE + 20_001, BASE], colour)
    assert p.returncode == 1
    assert "above the baseline by more than 2.0% (a regression)" in p.stderr
    assert "puct_run sims_256: 1020001" in p.stderr
    assert "gungraun bless" not in p.stderr


@pytest.mark.parametrize("colour", [False, True], ids=["plain", "colour"])
def test_gungraun_band_a_missing_case_fails_closed(tmp_path, colour):
    p = _band(tmp_path, [BASE, BASE], colour)
    assert p.returncode == 1
    assert "read 2 compared cases, expected 3" in p.stderr


def test_gungraun_band_a_case_without_a_baseline_fails_closed(tmp_path):
    log = tmp_path / "gungraun.log"
    log.write_text(_log([BASE, BASE, BASE], False).replace(f"{BASE}|{BASE}", f"{BASE}|N/A", 1))
    p = subprocess.run(["bash", str(BAND), str(log), "2.0", "3"], capture_output=True, text=True, check=False)
    assert p.returncode == 1
    assert "read 2 compared cases, expected 3" in p.stderr


CRATES = ["games_instructions", "tessera_rl_games"]


def _rustc(crate, flags, colour=False):
    running = "\x1b[1m\x1b[32m     Running\x1b[0m" if colour else "     Running"
    return (f"{running} `CARGO=/usr/local/cargo/bin/cargo /usr/local/rustup/toolchains/x/bin/rustc --crate-name {crate} "
            f"--edition=2024 src/lib.rs --crate-type lib -C opt-level=3 {flags} -C debuginfo=0`")


def _profile(tmp_path, lines):
    log = tmp_path / "build.log"
    log.write_text("\n".join(["   Compiling tessera-rl-games v0.1.0", *lines]) + "\n")
    return subprocess.run(["bash", str(PROFILE), str(log), *CRATES], capture_output=True, text=True, check=False)


@pytest.mark.parametrize("colour", [False, True], ids=["plain", "colour"])
def test_gungraun_profile_one_unit_without_lto_passes(tmp_path, colour):
    p = _profile(tmp_path, [_rustc(c, "-C codegen-units=1", colour) for c in CRATES])
    assert p.returncode == 0, p.stderr


def test_gungraun_profile_default_units_fails(tmp_path):
    p = _profile(tmp_path, [_rustc("games_instructions", "-C codegen-units=1"), _rustc("tessera_rl_games", "-C embed-bitcode=no")])
    assert p.returncode == 1
    assert "tessera_rl_games was compiled without -C codegen-units=1" in p.stderr


def test_gungraun_profile_codegen_units_16_fails(tmp_path):
    p = _profile(tmp_path, [_rustc(c, "-C codegen-units=16") for c in CRATES])
    assert p.returncode == 1
    assert "without -C codegen-units=1" in p.stderr


def test_gungraun_profile_fat_lto_fails(tmp_path):
    p = _profile(tmp_path, [_rustc("games_instructions", "-C lto=fat -C codegen-units=1"), _rustc("tessera_rl_games", "-C codegen-units=1")])
    assert p.returncode == 1
    assert "games_instructions was compiled with LTO" in p.stderr


def test_gungraun_profile_missing_rustc_lines_fail_closed(tmp_path):
    p = _profile(tmp_path, ["       Fresh tessera-rl-games v0.1.0"])
    assert p.returncode == 1
    assert "no rustc line for games_instructions" in p.stderr


def test_gungraun_profile_a_crate_whose_name_extends_a_listed_one_does_not_count(tmp_path):
    p = _profile(tmp_path, [_rustc("games_instructions_extra", "-C codegen-units=1"), _rustc("tessera_rl_games", "-C codegen-units=1")])
    assert p.returncode == 1
    assert "no rustc line for games_instructions" in p.stderr


def test_gungraun_gate_builds_and_runs_with_the_gate_profile():
    text = GATE.read_text()
    assert "profile=bench-gate" in text
    assert "band=2.0" in text
    runs = re.findall(r"cargo bench --locked (.*?) \"\$\{benches\[@\]\}\"", text)
    assert runs and all('--profile "$profile"' in r for r in runs), runs
    assert "CARGO_TERM_COLOR=never cargo bench --locked --profile" in text
    assert '--callgrind-limits="ir=${band}%"' in text
    assert text.count("build_with_profile") == 3  # defined, then called by check and by bless
    assert '"$root/scripts/gungraun_band.sh" "$log" "$band" "${#expected[@]}"' in text


def test_gungraun_gate_profile_is_one_unit_without_lto():
    cargo = (ROOT / "Cargo.toml").read_text()
    section = cargo.split("[profile.bench-gate]\n", 1)[1].split("\n[", 1)[0]
    settings = [ln for ln in section.splitlines() if ln and not ln.startswith("#")]
    assert settings == ['inherits = "bench"', "codegen-units = 1"]


def test_gungraun_gate_manifest_records_the_profile_settings():
    text = GATE.read_text()
    body = re.search(r"^profile_settings\(\) \{\n(.*?)\n\}", text, re.M | re.S).group(1)
    script = f'root="{ROOT}"; profile=bench-gate; {body}'
    out = subprocess.run(["bash", "-c", script], capture_output=True, text=True, check=True).stdout.strip()
    assert out == "inherits=bench;codegen-units=1"
    assert 'bench_profile = "$profile: $(profile_settings)"' in text
