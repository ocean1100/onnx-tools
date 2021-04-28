import argparse
import onnx
import numpy as np

def parse_args():
    parser=argparse.ArgumentParser(description='merge one ONNX model to another by mapping input tensors names')
    parser.add_argument('model_to', type=str, default='', help='input ONNX model to be merged to')
    parser.add_argument('model_in', type=str, default='', help='input ONNX model')
    parser.add_argument('model_out', type=str, default='', help='output ONNX model')
    parser.add_argument('-m', '--map', type=str, nargs='+', default=[],
            help='input tensor names mapping, in [tensor name in model_in]:[tensor name in model_to] format')

    args=parser.parse_args()
    return args

if __name__=='__main__':
    args=parse_args()
    mapping=dict()
    for item in args.map:
        src, dst=item.split(':')
        mapping[src]=dst

    model_to=onnx.load(args.model_to)
    model_in=onnx.load(args.model_in)
    assert model_to.ir_version==model_in.ir_version

    model_to.graph.initializer.extend(model_in.graph.initializer)
    model_to.graph.output.extend(model_in.graph.output)
    model_to.graph.value_info.extend(model_in.graph.value_info)
    for item in model_in.graph.node:
        for i in range(len(item.input)):
            if item.input[i] in mapping:
                item.input[i]=mapping[item.input[i]]
        model_to.graph.node.append(item)
    for item in model_in.graph.input:
        if item.name not in mapping:
            model_to.graph.input.append(item)
    model_to=onnx.shape_inference.infer_shapes(model_to)
    onnx.save(model_to, args.model_out)
