 PYTORCH_ENABLE_MPS_FALLBACK=1 CUDA_VISIBLE_DEVICES=0 python train_mimic3.py --task ihm --model_name copula --devices 1 \
 --lamb_copula 0.00001 --first_nrows 50