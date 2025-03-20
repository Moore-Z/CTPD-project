 CUDA_VISIBLE_DEVICES=0 python train_mimic3.py --task ihm --model_name ctpd --devices 1 \
    --use_multiscale --lamb1 0.1 --lamb2 0.5 --lamb3 0.5 
