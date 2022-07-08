## Teacher-free Feature Distillation

### Self-Regulated Feature Learning via Teacher-free Feature Distillation

Keywords: Knowledge Distillation, Feature Regularization

[paper](https://github.com/lilujunai/Tf-FD.github.io/edit/gh-pages/index.md), [code](https://github.com/lilujunai/Tf-FD.github.io/edit/gh-pages/index.md), [log&model](https://github.com/lilujunai/Tf-FD.github.io/edit/gh-pages/index.md),

### Abstract

Feature distillation is a widely used knowledge distillation approach, which always leads to significant performance improvements. Conventional feature distillation framework demands extra training budgets of teachers and complex transformations to align the features between teacher-student models. To address the problem, we analyze teacher roles in feature distillation and have an intriguing observation: additional teacher architectures are not always necessary. Then we propose TfFD, a simple and effective Teacher-free Feature Distillation framework, which seeks to reuse the privileged features within the student network itself. Specifically, we firstly present inter-layer Tf-FD, which squeezes feature knowledge in the deeper layers into the shallow ones by minimizing feature losses. Secondly, saliency features are leveraged to distill redundant features on the same layer for the intra-layer Tf-FD. Thanks to the narrow gap of these self-features, Tf-FD only needs to adopt a simple l2 loss without complex transformations. Meanwhile, Tf-FD is generic and can be directly deployed for training deep neural networks like most feature regularization methods. Furthermore, we provide insightful discussions about Tf-FD from regularization perspectives. In experiments, Tf-FD achieves superior performance and at least 3× faster training speed than state-of-the-art teacher-based feature distillation methods. Without any extra inference cost, Tf-FD obtains 0.91% and 1.84% top-1 accuracy gains for ResNet18 and ResNet50 on ImageNet, which substantially outperforms other regularization methods. Code will be made publicly available


Whenever you commit to this repository, GitHub Pages will run [Jekyll](https://jekyllrb.com/) to rebuild the pages in your site, from the content in your Markdown files.

### Markdown

Markdown is a lightweight and easy-to-use syntax for styling your writing. It includes conventions for

```markdown
Syntax highlighted code block

# Header 1
## Header 2
### Header 3

- Bulleted
- List

1. Numbered
2. List

**Bold** and _Italic_ and `Code` text

[Link](url) and ![Image](src)
```

For more details see [Basic writing and formatting syntax](https://docs.github.com/en/github/writing-on-github/getting-started-with-writing-and-formatting-on-github/basic-writing-and-formatting-syntax).

### Jekyll Themes

Your Pages site will use the layout and styles from the Jekyll theme you have selected in your [repository settings](https://github.com/lilujunai/Tf-FD.github.io/settings/pages). The name of this theme is saved in the Jekyll `_config.yml` configuration file.

### Support or Contact

Having trouble with Pages? Check out our [documentation](https://docs.github.com/categories/github-pages-basics/) or [contact support](https://support.github.com/contact) and we’ll help you sort it out.
