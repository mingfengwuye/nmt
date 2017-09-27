#!/bin/bash

python -m nmt.nmt \
    --src=zh --tgt=en --vocab_prefix=./temp/nmt_problem/tmp/vocab --train_prefix=./temp/nmt_problem/tmp/dev3 --dev_prefix=./temp/nmt_problem/tmp/dev3 --test_prefix=./temp/nmt_problem/tmp/dev3 --out_dir=./temp/nmt_problem/nmt_model_tmp --num_train_steps=12000 --steps_per_stats=100 --num_layers=2 --num_units=128 --dropout=0.2 --metrics=bleu
