CUDA_VISIBLE_DEVICES=0 python -u train.py --save_root "./results/tfd/"  --t_model results/base/base-c100-r110/initial_r110.pth.tar --s_init results/base/base-c100-r20/initial_r20.pth.tar --data_name cifar100 --num_class 100 --t_name resnet110  --s_name resnet20  --lambda_kd 0 --lambda_intra 4e-4 --kd-warm-up 1 --lambda_inter 5e-3  --note tfd+-c10-r110-r20-intra-4e-4-inter-5e-3-warmup-1-exp5