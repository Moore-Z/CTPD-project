CUDA_VISIBLE_DEVICES=0 python train_mimic3.py --devices 1 --task los --batch_size 128 --model_name ctpd \
    --use_multiscale --lamb1 0.1 --lamb2 0.5 --lamb3 0.5 --first_nrows 50