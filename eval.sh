#!/usr/bin/env bash

python3 eval.py \
    --budget 2500000 \
    --resume logs/level0_20210201-012820/model/final.pth \
    --viz \
    --prefix fulltrain \
    --video LSU-20200423 \
