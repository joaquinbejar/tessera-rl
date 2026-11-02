# SPDX-License-Identifier: MIT OR Apache-2.0
"""The provenance block every P2-M3 artifact carries: source, image,
cpuset, runtime versions, host and the hash of the tool's configuration.

The source SHA is a required argument of every tool. The image digest and
cpuset come from the environment the run's container sets
(``PILOT_IMAGE_DIGEST``, ``PILOT_CPUSET``), as in the pilot, and are
recorded as ``null`` when unset.
"""

from __future__ import annotations

import os
import platform
import socket

import artifacts
import numpy as np
import tessera_rl
import torch


def block(source_sha: str, config: dict) -> dict:
    return {
        "source_sha": source_sha,
        "image_digest": os.environ.get("PILOT_IMAGE_DIGEST"),
        "cpuset": os.environ.get("PILOT_CPUSET"),
        "host": socket.gethostname(),
        "versions": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "tessera_rl": tessera_rl.version(),
        },
        "config_sha256": artifacts.sha256(artifacts.canonical_json(config)),
    }
