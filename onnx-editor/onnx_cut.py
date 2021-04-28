import argparse
import onnx
import numpy as np

def parse_args():
    parser=argparse.ArgumentParser(description='cut ONNX model by input and output tensors names')
    parser.add_argument('model_in', type=str, default='', help='input ONNX model')
    parser.add_argument('model_out', type=str, default='', help='output ONNX model')
    parser.add_argument('-i', '--input', type=str, nargs='+', default=[],
            help='input tensor names, use input tensor in input model by default')
    parser.add_argument('-o', '--output', type=str, nargs='+', default=[],
            help='output tensor names, use output tensor in input model by default')

    args=parser.parse_args()
    return args

if __name__=='__main__':
    args=parse_args()
    model=onnx.load(args.model_in)
    if len(args.input)==0:
        for item in model.graph.input:
            args.input.append(item.name)
    if len(args.output)==0:
        for item in model.graph.output:
            args.output.append(item.name)
    onnx.utils.extract_model(args.model_in, args.model_out, args.input, args.output)
