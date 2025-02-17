#!/usr/bin bash
for seed in 1 186 196 2 0
do 
    python -u down.py \
        --dataset hwu64 \
        --save_model_path model_mtp \
        --seed $seed \
        --save_model 
done
