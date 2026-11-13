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
- mlp_connect4.onnx: a small float32 MLP (84 -> 32 -> 7 priors by softmax,
  and a tanh value [N, 1]) with weights from a seeded generator, for the
  same-machine repeatability test.
- slow_connect4.onnx: constant-like priors and a value from a chain of
  CHAIN 512x512 float32 matrix products per row, so one batch takes a
  measurable time (the test that inference releases the Python
  interpreter). Its matrices are built in the graph from one row each, so
  the file stays small.

Two fixtures come from spike S1's generator, not from this script:
s1_mlp.onnx and s1_fuse.onnx (their sha256 are checked by the tests).
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


def mlp_connect4():
    rng = np.random.default_rng(2026)
    def w(name, *shape):
        return numpy_helper.from_array((rng.standard_normal(shape) * 0.2).astype(np.float32), name)
    board = helper.make_tensor_value_info("board", TensorProto.UINT8, ["N", 2, 6, 7])
    priors = helper.make_tensor_value_info("priors", TensorProto.FLOAT, ["N", 7])
    value = helper.make_tensor_value_info("value", TensorProto.FLOAT, ["N", 1])
    flat = numpy_helper.from_array(np.array([-1, 84], dtype=np.int64), "flat")
    nodes = [
        helper.make_node("Cast", ["board"], ["x"], to=TensorProto.FLOAT),
        helper.make_node("Reshape", ["x", "flat"], ["x2"]),
        helper.make_node("Gemm", ["x2", "w1", "b1"], ["h"]),
        helper.make_node("Relu", ["h"], ["hr"]),
        helper.make_node("Gemm", ["hr", "w2", "b2"], ["logits"]),
        helper.make_node("Softmax", ["logits"], ["priors"], axis=1),
        helper.make_node("Gemm", ["hr", "w3", "b3"], ["v"]),
        helper.make_node("Tanh", ["v"], ["value"]),
    ]
    inits = [flat, w("w1", 84, 32), w("b1", 32), w("w2", 32, 7), w("b2", 7), w("w3", 32, 1), w("b3", 1)]
    graph = helper.make_graph(nodes, "tessera_onnx_mlp", [board], [priors, value], initializer=inits)
    m = helper.make_model(graph, producer_name="tessera-rl gen_fixtures.py",
                          opset_imports=[helper.make_opsetid("", OPSET)])
    m.ir_version = IR
    onnx.checker.check_model(m)
    return m


CHAIN = 400


def slow_connect4():
    board = helper.make_tensor_value_info("board", TensorProto.UINT8, ["N", 2, 6, 7])
    priors = helper.make_tensor_value_info("priors", TensorProto.DOUBLE, ["N", 7])
    value = helper.make_tensor_value_info("value", TensorProto.FLOAT, ["N", 1])
    starts = numpy_helper.from_array(np.array([0], dtype=np.int64), "starts")
    ends = numpy_helper.from_array(np.array([1], dtype=np.int64), "ends")
    one = numpy_helper.from_array(np.array([1], dtype=np.int64), "one")
    row7 = numpy_helper.from_array(np.full((1, 7), 1.0 / 7.0, dtype=np.float64), "row7")
    flat = numpy_helper.from_array(np.array([-1, 84], dtype=np.int64), "flat")
    w_in_row = numpy_helper.from_array(np.full((1, 512), 1.0 / 84.0, dtype=np.float32), "w_in_row")
    w_in_rep = numpy_helper.from_array(np.array([84, 1], dtype=np.int64), "w_in_rep")
    w_row = numpy_helper.from_array(np.full((1, 512), 1.0 / 512.0, dtype=np.float32), "w_row")
    w_rep = numpy_helper.from_array(np.array([512, 1], dtype=np.int64), "w_rep")
    nodes = [
        helper.make_node("Shape", ["board"], ["shape"]),
        helper.make_node("Slice", ["shape", "starts", "ends"], ["n"]),
        helper.make_node("Concat", ["n", "one"], ["tile"], axis=0),
        helper.make_node("Tile", ["row7", "tile"], ["priors"]),
        helper.make_node("Tile", ["w_row", "w_rep"], ["w"]),
        helper.make_node("Tile", ["w_in_row", "w_in_rep"], ["w_in"]),
        helper.make_node("Cast", ["board"], ["x"], to=TensorProto.FLOAT),
        helper.make_node("Reshape", ["x", "flat"], ["x2"]),
        helper.make_node("MatMul", ["x2", "w_in"], ["h0"]),
    ]
    for i in range(CHAIN):
        nodes.append(helper.make_node("MatMul", [f"h{i}", "w"], [f"h{i + 1}"]))
    nodes.append(helper.make_node("ReduceMean", [f"h{CHAIN}"], ["m"], axes=[1], keepdims=1))
    nodes.append(helper.make_node("Tanh", ["m"], ["value"]))
    graph = helper.make_graph(nodes, "tessera_onnx_slow", [board], [priors, value],
                              initializer=[starts, ends, one, row7, flat, w_in_row, w_in_rep, w_row, w_rep])
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
    onnx.save(mlp_connect4(), "mlp_connect4.onnx")
    onnx.save(slow_connect4(), "slow_connect4.onnx")
    if Path("external.bin").exists():
        Path("external.bin").unlink()
    onnx.save(model(), "external.onnx", save_as_external_data=True,
              all_tensors_to_one_file=True, location="external.bin", size_threshold=0)


if __name__ == "__main__":
    main(sys.argv[1])
