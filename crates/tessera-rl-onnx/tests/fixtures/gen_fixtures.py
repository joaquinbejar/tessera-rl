# SPDX-License-Identifier: MIT OR Apache-2.0
"""Generates the ONNX fixtures of tests/evaluator.rs (ADR-0057).

Usage: python3 gen_fixtures.py <out-dir>   (needs the `onnx` package; tested
with onnx 1.23.2). The outputs are deterministic: tests/evaluator.rs checks
each file's sha256, so a change here must update those hashes.

All models take Connect4's observation, input `board`, uint8 [N, 2, 6, 7]:
- constant.onnx: priors float64 [N, 7] = 1/7, value float64 [N] = 0.25.
  Both values are the float64 nearest to the real number, so a Rust
  evaluator writing 1.0 / 7.0 and 0.25 gives the same bits (the only
  cross-platform ONNX test).
- wrong_input.onnx: as constant, but the input is named `obs`.
- wrong_actions.onnx: as constant, but priors are [N, 6].
- external.onnx with external.bin: as constant, but the priors' constant
  row comes from an initializer stored in an external-data file.
"""

import os
import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

OPSET = 17
IR = 8


def model(input_name="board", actions=7, external=False):
    board = helper.make_tensor_value_info(input_name, TensorProto.UINT8, ["N", 2, 6, 7])
    priors = helper.make_tensor_value_info("priors", TensorProto.DOUBLE, ["N", actions])
    value = helper.make_tensor_value_info("value", TensorProto.DOUBLE, ["N"])
    starts = numpy_helper.from_array(np.array([0], dtype=np.int64), "starts")
    ends = numpy_helper.from_array(np.array([1], dtype=np.int64), "ends")
    one = numpy_helper.from_array(np.array([1], dtype=np.int64), "one")
    row = numpy_helper.from_array(np.full((1, actions), 1.0 / 7.0, dtype=np.float64), "row")
    nodes = [
        helper.make_node("Shape", [input_name], ["shape"]),
        helper.make_node("Slice", ["shape", "starts", "ends"], ["n"]),
        helper.make_node("Concat", ["n", "one"], ["tile"], axis=0),
        helper.make_node("Tile", ["row", "tile"], ["priors"]),
        helper.make_node(
            "ConstantOfShape",
            ["n"],
            ["value"],
            value=numpy_helper.from_array(np.array([0.25], dtype=np.float64)),
        ),
    ]
    graph = helper.make_graph(nodes, "tessera_onnx_fixture", [board], [priors, value],
                              initializer=[starts, ends, one, row])
    m = helper.make_model(graph, producer_name="tessera-rl gen_fixtures.py",
                          opset_imports=[helper.make_opsetid("", OPSET)])
    m.ir_version = IR
    onnx.checker.check_model(m)
    return m


def main(out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    # onnx resolves the external-data location against the working directory.
    os.chdir(out)
    onnx.save(model(), "constant.onnx")
    onnx.save(model(input_name="obs"), "wrong_input.onnx")
    onnx.save(model(actions=6), "wrong_actions.onnx")
    if Path("external.bin").exists():
        Path("external.bin").unlink()
    onnx.save(model(), "external.onnx", save_as_external_data=True,
              all_tensors_to_one_file=True, location="external.bin", size_threshold=0)


if __name__ == "__main__":
    main(sys.argv[1])
