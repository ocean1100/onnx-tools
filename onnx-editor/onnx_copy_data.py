import argparse
import onnx
import copy

def parse_args():
    parser=argparse.ArgumentParser(description='copy tensor data from one or more ONNX to another\'s structure, associate by name')
    parser.add_argument('model_in', type=str, default='', help='input ONNX model, for structure and initial tensor data')
    parser.add_argument('model_out', type=str, default='', help='output ONNX model')
    parser.add_argument('-d', '--data', type=str, nargs='+', help='input ONNX model, for copied tensor data')

    args=parser.parse_args()
    return args

def strip_tensor(tensor):
    for item in ['float_data', 'int32_data', 'int64_data', 'double_data', 'uint64_data', 'raw_data']:
        tensor.ClearField(item)

def copy_tensor_data(tensor_to, tensor_from):
    if tensor_to.dims!=tensor_from.dims or tensor_to.data_type!=tensor_from.data_type:
        raise RuntimeError('TensorProto metadata mismatch')
    for item in ['segment', 'data_location']:
        if tensor_to.HasField(item) or tensor_from.HasField(item):
            raise RuntimeError('TensorProto has unsupported field: %s'%(item))
    if len(tensor_to.external_data)!=0 or len(tensor_from.external_data)!=0:
        raise RuntimeError('TensorProto has unsupported field: external_data')
    strip_tensor(tensor_to)
    tensor_to.float_data[:]=tensor_from.float_data
    tensor_to.int32_data[:]=tensor_from.int32_data
    tensor_to.int64_data[:]=tensor_from.int64_data
    tensor_to.double_data[:]=tensor_from.double_data
    tensor_to.uint64_data[:]=tensor_from.uint64_data
    if tensor_from.HasField('raw_data'):
        tensor_to.raw_data=tensor_from.raw_data

def copy_model_data(model_to, model_from):
    name_to_initializer=dict()
    for item in model_from.graph.initializer:
        name_to_initializer[item.name]=item
    for item in model_to.graph.initializer:
        if item.name in name_to_initializer:
            copy_tensor_data(item, name_to_initializer[item.name])

    name_to_constant=dict()
    for item in model_from.graph.node:
        if item.op_type=='Constant':
            if item.attribute[0].name=='value':
                name_to_constant[item.name]=item.attribute[0].t
    for item in model_to.graph.node:
        if item.op_type=='Constant':
            if item.attribute[0].name=='value':
                if item.name in name_to_constant:
                    copy_tensor_data(item.attribute[0].t, name_to_constant[item.name])

if __name__=='__main__':
    args=parse_args()
    model=onnx.load(args.model_in)
    for item in args.data:
        copy_model_data(model, onnx.load(item))
    onnx.save(model, args.model_out)
