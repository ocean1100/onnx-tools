# ONNX Tools

## dependencies

python3

onnx

## usage 

please refer to each tools parse_args() function

## example

python3 onnx_cut.py in.onnx out.onnx -i foo -o bar baz

Cut submodel from in.onnx by using 'foo' as its input blob, 'bar'+'baz' as its output blobs, save submodel to out.onnx. An error will be raised if no valid graph is defined by given inputs / outputs (e.g. given outputs are not fully produced by given inputs).

python3 onnx_merge.py a.onnx b.onnx out.onnx -m foo:bar

Merge b.onnx into a.onnx and change input blob of all operators in b.onnx named 'foo' to 'bar', save merged model to out.onnx. If b.onnx contains other input blobs, they will appear in out.onnx as input blobs. Shape inference will be performed on merge model based on new graph topology. Users should make sure no operator / initializer / blob name conflicts between a.onnx and b.onnx, and merged model is valid.

python3 onnx_name_edit.py foo[0-9] bar in.onnx out.onnx --skip-io

Change all operators and blobs names (except input and output blobs) by matching regex expression 'foo[0-9]' and replace all matches to 'bar' in in.onnx, and save result to out.onnx.

python3 onnx_name_edit.py '^' bar_ in.onnx out.onnx

Add a prefix 'bar_' to all operators and blobs names in in.onnx, and save result to out.onnx.

python3 onnx_name_edit.py '$' _bar in.onnx out.onnx

Add a postfix '_bar' to all operators and blobs names in in.onnx, and save result to out.onnx.

python3 onnx_copy_data.py in.onnx out.onnx -d data1.onnx data2.onnx

Copy all data (initializers and constant operators) from data1.onnx and data2.onnx (in order) to in.onnx, associating by initializers' and operators' name, and save result to out.onnx.

python3 onnx_extract_constant.py in.onnx out.onnx -n foo

Extract constant operator produce blob named 'foo' to initializer in in.onnx, and save result to out.onnx.

python3 onnx_to_readable.py in.onnx out.txt -s

Load in.onnx and strip all tensor in initializers and constant operators, and save result with readable protobuf text format onnx model to out.txt. User can then open or edit out.txt with text editor.

python3 onnx_from_readable.py in.txt out.onnx

Load readable protobuf text format onnx model from in.txt, and save it to out.onnx.
