# SoftTarget
CUDA_VISIBLE_DEVICES=0 python -u train_kd.py --save_root "./results/tfd/" --t_model results/base/base-c100-r110/initial_r110.pth.tar  --s_init results/base/base-c100-r20/initial_r20.pth.tar --data_name cifar100 --num_class 100 --t_name resnet110 --s_name resnet20 --kd_mode tfd --lambda_kd 5e-3 --lambda_kd1 4e-4 --T 4.0 --note tfd-c10-r110-r20-intra-4e-4-inter-5e-3-warm-up-exp2
CUDA_VISIBLE_DEVICES=0 python -u train_kd.py --save_root "./results/tfd/" --t_model results/base/base-c100-r110/initial_r110.pth.tar  --s_init results/base/base-c100-r20/initial_r20.pth.tar --data_name cifar100 --num_class 100 --t_name resnet110 --s_name resnet20 --kd_mode tfd --lambda_kd 1e-2 --lambda_kd1 1e-3 --T 4.0 --note tfd-c10-r110-r20-intra-1e-3-inter-e-2-warm-up-exp1
CUDA_VISIBLE_DEVICES=1 python -u train_kd.py --save_root "./results/tfd/" --t_model results/base/base-c100-r110/initial_r110.pth.tar  --s_init results/base/base-c100-r20/initial_r20.pth.tar --data_name cifar100 --num_class 100 --t_name resnet110 --s_name resnet20 --kd_mode tfd --lambda_kd 1e-3 --lambda_kd1 1e-4 --T 4.0 --note tfd-c10-r110-r20-intra-1e-4-inter-1e-3-warm-up-exp1
CUDA_VISIBLE_DEVICES=1 python -u train_kd.py --save_root "./results/tfd/" --t_model results/base/base-c100-r110/initial_r110.pth.tar  --s_init results/base/base-c100-r20/initial_r20.pth.tar --data_name cifar100 --num_class 100 --t_name resnet110 --s_name resnet20 --kd_mode tfd --lambda_kd 5e-3 --lambda_kd1 5e-4 --T 4.0 --note tfd-c10-r110-r20-intra-5e-4-inter-5e-3-warm-up-exp2