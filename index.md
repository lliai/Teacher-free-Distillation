## Teacher-free Feature Distillation

### Self-Regulated Feature Learning via Teacher-free Feature Distillation

Keywords: Knowledge Distillation, Feature Regularization

[paper](https://github.com/lilujunai/Tf-FD.github.io/edit/gh-pages/index.md), [code](https://github.com/lilujunai/Teacher-free-Distillation), [Training logs & model](https://pan.baidu.com/s/1-1oKjctjSxzlWHygkffG_g),

### Abstract

Feature distillation is a widely used knowledge distillation approach, which always leads to significant performance improvements. Conventional feature distillation framework demands extra training budgets of teachers and complex transformations to align the features between teacher-student models. To address the problem, we analyze teacher roles in feature distillation and have an intriguing observation: additional teacher architectures are not always necessary. Then we propose TfFD, a simple and effective Teacher-free Feature Distillation framework, which seeks to reuse the privileged features within the student network itself. Specifically, we firstly present inter-layer Tf-FD, which squeezes feature knowledge in the deeper layers into the shallow ones by minimizing feature losses. Secondly, saliency features are leveraged to distill redundant features on the same layer for the intra-layer Tf-FD. Thanks to the narrow gap of these self-features, Tf-FD only needs to adopt a simple l2 loss without complex transformations. Meanwhile, Tf-FD is generic and can be directly deployed for training deep neural networks like most feature regularization methods. Furthermore, we provide insightful discussions about Tf-FD from regularization perspectives. In experiments, Tf-FD achieves superior performance and at least 3× faster training speed than state-of-the-art teacher-based feature distillation methods. Without any extra inference cost, Tf-FD obtains 0.91% and 1.84% top-1 accuracy gains for ResNet18 and ResNet50 on ImageNet, which substantially outperforms other regularization methods. Code will be made publicly available

### Bibtex 


```markdown
@inproceedings{li2022TfFD,
    title={Self-Regulated Feature Learning via Teacher-free Feature Distillation},
    author={Li, Lujun},
    booktitle={European Conference on Computer Vision (ECCV)},
    year={2022}
}
```


### Support or Contact

lilujunai@gmail.com
