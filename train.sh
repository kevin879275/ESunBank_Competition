python train.py \
    --batchsize 512 \
    --method efficientnet \
    --method_level b0 \
    -lr 1.6e-2 \
    -ending-lr 1.6e-4 \
    --use-padding True \
    --threshold 0.7 # -1 for not threshold