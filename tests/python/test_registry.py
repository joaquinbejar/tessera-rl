# SPDX-License-Identifier: MIT OR Apache-2.0
"""The built module lists every shipped environment (ADR-0054).

A crate whose registrations the linker dropped would be missing here; the
wheels W0 jobs run this file against each built wheel.
"""

from tessera_rl import _native

SHIPPED = {
    "tessera/Acrobot-v1": (1, 1, "cross-platform", True),
    "tessera/BrickWall-v1": (1, 1, "cross-platform", True),
    "tessera/CartPole-v1": (1, 1, "cross-platform", True),
    "tessera/Connect4-v1": (1, 2, "cross-platform", True),
    "tessera/Go9-v1": (1, 2, "cross-platform", True),
    "tessera/Grid-v1": (1, 1, "cross-platform", True),
    "tessera/GridKeyDoor-v1": (1, 4, "cross-platform", True),
    "tessera/Leduc-v1": (1, 2, "cross-platform", False),
    "tessera/MountainCar-v1": (1, 1, "cross-platform", True),
    "tessera/Othello-v1": (1, 2, "cross-platform", True),
    "tessera/PaddleDuel-v1": (1, 2, "cross-platform", True),
    "tessera/Pendulum-v1": (1, 1, "cross-platform", True),
    "tessera/RTS-v1": (2, 2, "cross-platform", False),
    "tessera/Snake-v1": (1, 2, "cross-platform", True),
    "tessera/Tag-v1": (1, 8, "cross-platform", False),
}


def test_registered_envs_every_shipped_environment_listed():
    got = _native.registered_envs()
    assert [row[0] for row in got] == sorted(SHIPPED)
    assert {row[0]: tuple(row[1:]) for row in got} == SHIPPED


def test_registered_envs_rows_are_typed_tuples():
    for row in _native.registered_envs():
        assert isinstance(row, tuple) and len(row) == 5
        id_, version, max_agents, tier, perfect = row
        assert isinstance(id_, str) and isinstance(tier, str)
        assert isinstance(version, int) and isinstance(max_agents, int)
        assert isinstance(perfect, bool)
