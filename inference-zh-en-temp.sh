#!/bin/bash

python -m nmt.nmt --src=zh --tgt=en --model_dir=./temp/nmt_problem/nmt_model_tmp --vocab_prefix=./temp/nmt_problem/tmp --out_dir=./temp/nmt_problem/nmt_model_tmp --inference_input_file=./temp/nmt_problem/tmp/dev2.zh  --inference_output_file=./temp/nmt_problem/nmt_model_tmp/result

echo "Result file is ./temp/nmt_problem/nmt_model_tmp/result"
