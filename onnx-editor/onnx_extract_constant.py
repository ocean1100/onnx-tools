import argparse
import onnx
import numpy as np

def parse_args():
    parser=argparse.ArgumentParser(description='extract constant operator in ONNX model to initializer by blob names')
    parser.add_argument('model_in', type=str, default='', help='input ONNX model')
    parser.add_argument('model_out', type=str, default='', help='output ONNX model')
    parser.add_argument('-n', '--name', type=str, nargs='+', default=[],
            help='blob names, use all blobs produced by constant operator by default')

    args=parser.parse_args()
    return args

if __name__=='__main__':
    args=parse_args()
    model=onnx.load(args.model_in)
    model_out=onnx.ModelProto()
    model_out.CopyFrom(model)
    model_out.graph.ClearField('node')
    for item in model.graph.node:
        if item.op_type=='Constant' and (item.output[0] in args.name or len(args.name)==0):
            initializer=onnx.TensorProto()
            if item.attribute[0].name=='value':
                initializer.CopyFrom(item.attribute[0].t)
            else:
                raise RuntimeError('Constant operator use unsupported attribute: %s'%(item.attribute[0].name))
            initializer.name=item.output[0]
            model_out.graph.initializer.append(initializer)
        else:
            model_out.graph.node.append(item)
    onnx.save(model_out, args.model_out)
