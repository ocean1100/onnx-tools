import argparse
import onnx
import re

def parse_args():
    parser=argparse.ArgumentParser(description='ONNX model name edit')
    parser.add_argument('pattern', type=str, default='', help='regex pattern to match')
    parser.add_argument('replace', type=str, default='', help='replacement of matched pattern')
    parser.add_argument('model_in', type=str, default='', help='input model')
    parser.add_argument('model_out', type=str, default='', help='output model')
    parser.add_argument('--skip-node', action='store_true', help='skip node name')
    parser.add_argument('--skip-blob', action='store_true', help='skip blob (including initializer, excluding input / output) name')
    parser.add_argument('--skip-io', action='store_true', help='skip input / output blob name')

    args=parser.parse_args()
    return args

def process_node(graph, pattern, replace):
    for item in graph.node:
        item.name=re.sub(pattern, replace, item.name)

def update_node_io(graph, old_to_new_name):
    def update_list(name_list):
        for i in range(len(name_list)):
            if name_list[i] in old_to_new_name:
                name_list[i]=old_to_new_name[name_list[i]]
    
    for item in graph.node:
        update_list(item.input)
        update_list(item.output)

def process_blob(graph, pattern, replace):
    old_to_new_name=dict()
    for item in list(graph.value_info)+list(graph.initializer):
        new_name=re.sub(pattern, replace, item.name)
        old_to_new_name[item.name]=new_name
        item.name=new_name
    update_node_io(graph, old_to_new_name)

def process_io(graph, pattern, replace):
    old_to_new_name=dict()
    for item in list(graph.input)+list(graph.output):
        new_name=re.sub(pattern, replace, item.name)
        old_to_new_name[item.name]=new_name
        item.name=new_name
    update_node_io(graph, old_to_new_name)

if __name__=='__main__':
    args=parse_args()
    model=onnx.load(args.model_in)
    if not args.skip_node:
        process_node(model.graph, args.pattern, args.replace)
    if not args.skip_blob:
        process_blob(model.graph, args.pattern, args.replace)
    if not args.skip_io:
        process_io(model.graph, args.pattern, args.replace)
    onnx.save(model, args.model_out)
