## Teacher-free Feature Distillation

### Self-Regulated Feature Learning via Teacher-free Feature Distillation

Keywords: Knowledge Distillation, Feature Regularization

[paper](https://github.com/lilujunai/Tf-FD.github.io/edit/gh-pages/index.md), [code](https://github.com/lilujunai/Teacher-free-Distillation), [Training logs & model](https://pan.baidu.com/s/1-1oKjctjSxzlWHygkffG_g),

### Abstract

Knowledge distillation from intermediate feature representations always leads to significant performance improvements. Conventional feature distillation framework demands extra selecting/training budgets of teachers and complex transformations to align the features between teacher-student models. To address the problem, we analyze teacher roles in feature distillation and have an intriguing observation: additional teacher architectures are not always necessary. Then we propose TfFD, a simple and effective Teacher-free Feature Distillation framework seeking to reuse privileged features within the student network itself to provide teacher-like knowledge without an additional model. In particular, our framework is subdivided into inter-layer and intra-layer distillation. Firstly, inter-layer Tf-FD deals with distilling high-level semantic knowledge embedded in the deeper layers representations to guide the training of shallow layers. Secondly, intra-layer Tf-FD performs feature salience ranking and transfers the knowledge from salient feature to redundant feature within the same layer. Thanks to the narrow gap of these self-features, Tf-FD only needs to optimize simple feature mimicking losses without complex transformations. Furthermore, Tf-FD is generic and can be directly deployed for training deep models like other feature regularizers and we provide insightful discussions about Tf-FD from regularization perspectives. Extensive experiments on classification and object detection tasks demonstrate that our technique achieves state-of-the-art results on different models with fast training speed. Additionally, our method can be readily combined with common knowledge distillation strategies at the network heads to get further improved results. 

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
