# Teacher-free Feature Distillation
***This repo is in the process of re-building based on KD-ZOO. The final version will contain most of the models/exp in paper. Please stay tuned.***

This project provides Pytorch implementation for [Self-Regulated Feature Learning via Teacher-free Feature Distillation](https://lilujunai.github.io/Teacher-free-Distillation/).

## Requirements
```
` Python == 3.7, PyTorch == 1.3.1`
```

## Datasets
CIFAR-100 and Imagenet.


## Training
Run train_kd.py for training Tf-FD in CIFAR datasets. 

Tf-FD:


`python -u train_kd.py --save_root "./results/tfd/" --kd_mode tfd --lambda_inter 0.0005 --lambda_intra 0.0008 --note tfd-r20-inter-0.0005-intra-0.0008`


Tf-FD+(Tf-FD):


`python -u train_kd.py --save_root "./results/tfd+/" --kd_mode tfd+ --lambda_inter 0.0005 --lambda_intra 0.0008 --note tfd+-r20-inter-0.0005-intra-0.0008`


## Results
Most pretrained models and logs has been released on Baidu Netdisk:

link: https://pan.baidu.com/s/1-1oKjctjSxzlWHygkffG_g

pwd: i4fj

## Acknowledgements
This repo is partly based on the following repos, thank the authors a lot.
- [HobbitLong/RepDistiller](https://github.com/HobbitLong/RepDistiller)
- [Knowledge-Distillation-Zoo](https://github.com/AberHu/Knowledge-Distillation-Zoo)

## Citation
If you find that this project helps your research, please consider citing some of the following papers:

```
@inproceedings{li2022TfFD,
    title={Self-Regulated Feature Learning via Teacher-free Feature Distillation},
    author={Li, Lujun},
    booktitle={European Conference on Computer Vision (ECCV)},
    year={2022}
}
```

