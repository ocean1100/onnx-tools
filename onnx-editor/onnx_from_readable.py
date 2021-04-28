import argparse
import onnx
import google.protobuf.text_format

def parse_args():
    parser=argparse.ArgumentParser(description='ONNX from readable format')
    parser.add_argument('readable_in', type=str, default='', help='input readable text')
    parser.add_argument('model_out', type=str, default='', help='output ONNX model')

    args=parser.parse_args()
    return args

if __name__=='__main__':
    args=parse_args()
    model=google.protobuf.text_format.Parse(open(args.readable_in).read(), onnx.ModelProto())
    onnx.save(model, args.model_out)
