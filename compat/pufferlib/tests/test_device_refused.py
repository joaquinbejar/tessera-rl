# SPDX-License-Identifier: MIT OR Apache-2.0
"""The PufferLib adapter never passes ``device=`` through to ``make``."""

import pytest

import tessera_rl
from tessera_rl import puffer


@pytest.mark.parametrize("cls", [puffer.TesseraPufferEnv, puffer.TesseraPufferPipelinedEnv])
def test_puffer_device_refused(cls):
    with pytest.raises(tessera_rl.CapabilityError, match="PufferLib adapter"):
        cls("tessera/CartPole-v1", num_envs=2, device="cuda:0")
