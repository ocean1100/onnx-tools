import argparse
import onnx
import numpy as np

def parse_args():
    parser=argparse.ArgumentParser(description='ONNX to readable text format')
    parser.add_argument('model_in', type=str, default='', help='input ONNX model')
    parser.add_argument('readable_out', type=str, default='', help='output readable text')
    parser.add_argument('-s', '--strip-tensor', action='store_true',
            help='strip all tensor\'s data to reduce text size')

    args=parser.parse_args()
    return args

def strip_tensor(tensor):
    for item in ['float_data', 'int32_data', 'int64_data', 'double_data', 'uint64_data', 'raw_data']:
        tensor.ClearField(item)

def process_raw_data(tensor):
    if not tensor.HasField('raw_data'):
        return
    if tensor.data_type==onnx.TensorProto.FLOAT:
        tensor.float_data[:]=list(np.frombuffer(tensor.raw_data, dtype=np.float32))
    elif tensor.data_type==onnx.TensorProto.DOUBLE:
        tensor.double_data[:]=list(np.frombuffer(tensor.raw_data, dtype=np.float64))
    elif tensor.data_type==onnx.TensorProto.COMPLEX64:
        tensor.float_data[:]=list(np.frombuffer(tensor.raw_data, dtype=np.float32))
    elif tensor.data_type==onnx.TensorProto.COMPLEX128:
        tensor.double_data[:]=list(np.frombuffer(tensor.raw_data, dtype=np.float64))
    elif tensor.data_type==onnx.TensorProto.UINT8:
        tensor.int32_data[:]=list(np.frombuffer(tensor.raw_data, dtype=np.uint8))
    elif tensor.data_type==onnx.TensorProto.INT8:
        tensor.int32_data[:]=list(np.frombuffer(tensor.raw_data, dtype=np.int8))
    elif tensor.data_type==onnx.TensorProto.UINT16:
        tensor.int32_data[:]=list(np.frombuffer(tensor.raw_data, dtype=np.uint16))
    elif tensor.data_type==onnx.TensorProto.INT16:
        tensor.int32_data[:]=list(np.frombuffer(tensor.raw_data, dtype=np.int16))
    elif tensor.data_type==onnx.TensorProto.INT32:
        tensor.int32_data[:]=list(np.frombuffer(tensor.raw_data, dtype=np.int32))
    elif tensor.data_type==onnx.TensorProto.INT64:
        tensor.int64_data[:]=list(np.frombuffer(tensor.raw_data, dtype=np.int64))
    elif tensor.data_type==onnx.TensorProto.UINT32:
        tensor.uint64_data[:]=list(np.frombuffer(tensor.raw_data, dtype=np.uint32))
    elif tensor.data_type==onnx.TensorProto.UINT64:
        tensor.uint64_data[:]=list(np.frombuffer(tensor.raw_data, dtype=np.uint64))
    else:
        raise RuntimeError('Unknown tensor data type: %d'%(tensor.data_type))
    tensor.ClearField('raw_data')
    
if __name__=='__main__':
    args=parse_args()
    model=onnx.load(args.model_in)
    for item in model.graph.initializer:
        if args.strip_tensor:
            strip_tensor(item)
        else:
            process_raw_data(item)
    for item in model.graph.node:
        if item.op_type=='Constant':
            if item.attribute[0].name=='value':
                if args.strip_tensor:
                    strip_tensor(item.attribute[0].t)
                else:
                    process_raw_data(item.attribute[0].t)
    with open(args.readable_out, 'w') as f:
        f.write(str(model))
