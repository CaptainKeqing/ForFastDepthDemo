import tvm
import numpy as np
import argparse
import os
from tvm.contrib import graph_runtime


def init_model(model_dir, cuda=True):
    """Initiates the model and returns run, set_input, get_input functions"""
    # import compiled graph
    print("=> [TVM on myCom] using model files in {}".format(model_dir))
    assert(os.path.isdir(model_dir))

    print("=> [TVM on myCom] loading model lib and ptx")
    loaded_lib = tvm.runtime.load_module(os.path.join(model_dir, "mod.so"))

    print("=> [TVM on myCom] loading model graph and params")
    loaded_graph = open(os.path.join(model_dir,"mod.json")).read()
    loaded_params = bytearray(open(os.path.join(model_dir, "mod.params"), "rb").read())
    print("=> [TVM on myCom] creating TVM runtime module")

    fcreate = graph_runtime.create
    ctx = tvm.cuda(0) if cuda else tvm.cpu(0)
    print(ctx.device_type)
    gmodule = fcreate(loaded_graph, loaded_lib, ctx)
    set_input, get_output, run = gmodule["set_input"], gmodule["get_output"], gmodule["run"]

    print("=> [TVM on myCom] feeding params into TVM module")

    gmodule["load_params"](loaded_params)

    return run, set_input, get_output


def run_model(run, set_input, get_output, frame):
    set_input(0, tvm.nd.array(frame))
    run()
    out_shape = (1, 1, 224, 224)
    out = tvm.nd.empty(out_shape, "float32")
    get_output(0, out)
    return out.numpy()


