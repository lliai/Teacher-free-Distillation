## Teacher-free Feature Distillation

### Self-Regulated Feature Learning via Teacher-free Feature Distillation

Keywords: Knowledge Distillation, Feature Regularization

[paper](https://github.com/lilujunai/Tf-FD.github.io/edit/gh-pages/index.md), [code](https://github.com/lilujunai/Teacher-free-Distillation), [Training logs & model](https://pan.baidu.com/s/1-1oKjctjSxzlWHygkffG_g),



Feature distillation always leads to significant performance improvements, but requires extra training budgets. To address the problem, we propose TFD, a simple and effective \textbf{T}eacher-\textbf{F}ree \textbf{D}istillation framework, which seeks to reuse the privileged features within the student network itself. Specifically, TFD squeezes feature knowledge in the deeper layers into the shallow ones by minimizing feature loss. Thanks to the narrow gap of these self-features, TFD only needs to adopt a simple $l_2$ loss without complex transformations. Extensive experiments on recognition benchmarks show that our framework can achieve superior performance than teacher-based feature distillation methods. On the ImageNet dataset, our approach achieves $0.8\%$ gains for ResNet18, which surpasses other state-of-the-art training techniques.
\end{abstract}



\begin{IEEEkeywords}
Knowledge distillation
\end{IEEEkeywords}

%This demonstrates that feature distillation does not depend on additional teacher architectures.

\begin{figure*}
\centering
\includegraphics[width=0.8\textwidth]{IJCNN22-822-CR/inter.pdf}
\caption{Illustration of various distillation methods, including teacher selection framework~(a), teacher generation framework~(b), our depth-wise TFD~(c) and width-wise TFD~(d).} %We use ResNet32 as student model on CIFAR-100. Different teacher architectures, pre-trained ResNet110/ResNet32, online ResNet110/ResNet32 in (a) and auxiliary branch of ResNet32 in (b) improve baseline  by 1.36\%, 1.25\%, 1.63\%, 1.52\% and 0.92\% gains for top-1 accuracy, respectively. Our depth-wise TFD and width-wise TFD obtain 1.47\% and 1.17\% gains.}
  \label{fig:introduction}

\end{figure*}


\section{Introduction}
Alongside deep learning's tremendous success in different tasks~\cite{ref01_alexnet, ref02_rcnn,Attention,GPT-3},  it remains difficult to apply deep neural networks to real-world problems due to computational and memory constraints. To avoid this problem, many attempts~\cite{ref08_dorefa-net,ref08_inq,ref05_filter_pruning,ref04_weights_pruning} have been made to reduce the computational cost of deep learning models, with Knowledge Distillation (KD)~\cite{ref10_kd} being one of them. KD is a network training strategy that works by transferring knowledge from a high-capacity teacher model to a low-capacity target student model at runtime, resulting in a better accuracy-efficiency trade-off.

The logit output of the teacher network is used as knowledge in the original KD~\cite{ref10_kd}. The feature distillation methods~\cite{ref11_feature_kd,ref12_attention_kd} promote a student network to replicate the intermediate feature of a teacher network to further leverage the knowledge. Following research, such as~\cite{ref12_attention_kd, ref28_factor, ref43_vid, ref41_sp, AB, ref40_nst}, focus on extracting and matching informative knowledge conditioned on intermediate feature representations of a preset teacher model. However, there are three major flaws in the teacher-student knowledge distillation pipeline: (1) Finding appropriate teacher models, especially for big student models, necessitates significant effort and experimentation. (2) The training teacher model requires additional training resources, which adds to the application burden. (3) Teacher-based distillation methods always employ some training techniques~(\eg, early stopping~\cite{Cho2019OnTE}, decay~\cite{Zhou2020ChannelDC} and  complex feature transformations~\cite{ref28_factor,Overhaul} to perform better semantic alignment~\cite{Cross-Layer,ShowAA} due to the feature gap. These drawbacks limit the practical usages of feature distillation.


Naturally, the issue arises: is a second teacher model required for feature distillation? To demonstrate this, we look into the behaviors of teacher models in teacher-based distillation approaches, such as teacher selection and generation. As demonstrated in Figure~\ref{fig:introduction}, another high capacity model is frequently chosen as the instructor model for teacher selection methods. Meanwhile, teacher generation methods generate auxiliary branches that share the shallow layer with the student model to obtain the instructor model. As a result, in these two frameworks, the teacher-student model can be viewed as a hypernetwork with a teacher branch and a student branch. For the teacher branch, we analyze various sorts of models to see how they affect feature distillation. The results show that all of these different teaching models can result in significant distillation increases. To put it another way, subnetworks in different branches of the hypernetwork can act as teacher models for other subnetworks. This observation motivates us to investigate whether distillation enhancements may be achieved using features from subnetworks in other dimensions. As a result, we reject the teacher branch and distill using features found in various layers and channels (see Figure~\ref{fig:introduction} (c)). Surprisingly, a feature distillation strategy with no teachers provides considerable performance gains.


We provide a simple and effective \textbf{T}eacher-free \textbf{D}istillation~(TFD) framework based on the observations described above. Unlike the present teacher-student framework, our approach offers the use of privileged features inside the student network itself as a type of emphasis-free supervision to execute distillation without the use of supplementary instructor models for the first time. The depth-wise TFD causes the preceding blocks to resemble the properties of deeper blocks, which carry rich contextual information~\cite{ExplainingKD}. Thus, there is no need to select or create additional teacher architectures for TFD, which can be flexibly applied to various models. Then, TFD proposes a basic distillation pipeline that can efficiently broaden the use of distillation methods without requiring additional teacher training expenses. Because of the small semantic gap between self-features, TFD simply has to apply basic $l_2$ distances for the feature mimicking loss.


To illustrate the superiority of the proposed method, extensive experiments are performed on different deep models and datasets. Our method achieves gains in absolute accuracy of $0.74\%\sim 1.60\%$ and outperforms other regularization methods with obvious margins of $0.61\% \sim1.23\%$. At the same time, TFD achieves more than $3\times$ faster training speed than feature distillation methods with a high-capacity teacher model. On the ImageNet dataset, our approach achieves $0.8\%$ gains for ResNet18, which surpasses other state-of-the-art training techniques.



In summary, we make the following principle contributions in this paper:

\begin{itemize}

\item We demonstrate that the distillation process does not rely on additional teacher architectures by analyzing and exploring teacher models in feature distillation. This inspires us to propose a new Teacher-Free Feature Distillation (TFD) framework, which has yet to be achieved in the field of knowledge distillation.

\item TFD utilizes privileged self-features to perform distillation without the need for a separate teacher model or complicated transformations. On a variety of deep models and datasets, TFD outperforms other state-of-the-art feature distillation methods and accelerates training.

%\item We further discuss the relationship between TFD and feature regularization. TFD implicitly utilizes self-features as regularization distortion by optimizing the distillation loss. We hope this discussion could facilitate future research for feature distillation works to some extent.

\end{itemize}



 \begin{figure*}
  \centering
  \includegraphics[width=0.8\textwidth]{IJCNN22-822-CR/main.pdf}
  \caption{A schematic overview of our TFD, including depth-wise and width-wise parts. In the training phase, TFD capitalizes on features in deeper layers to supervise shallow ones using $l_2$-loss. In the inference phase, the model can be inferred separately.
}
  \label{fig:main}
\end{figure*}





\section{Related Work}
The main idea behind Knowledge Distillation (KD) is to use knowledge~(\eg, logits~\cite{ref10_kd,ref15_dml}, feature values~\cite{ref11_feature_kd} from a high-capacity teacher to guide the training of a student model. Apart from the ground truth labels, early pioneering works~\cite{ref25_first_kt, ref10_kd} utilize soft logit outputs of the pre-trained instructor as additional supervision to guide the student's training. A network's intermediate features provide a wealth of spatial and structural information on picture content~\cite{AFD}. To encourage the student model to emulate the feature representations of the teacher model, feature distillation approaches~\cite{ref11_feature_kd,ref26_gift,ref12_attention_kd, AFD} are presented. Because the dimensions of feature maps from different layers of student and teacher networks are frequently misaligned (e.g., widths, heights, and channels), existing feature distillation approaches use a variety of transformations to match their dimensions and different distance metrics to measure differences. For example, FitNets~\cite{ref11_feature_kd} mimics the intermediate features between the teacher and student networks with $l_2$-loss, and AT~\cite{ref12_attention_kd} applies feature distillation on the attention map. However, selecting a suitable teacher model for feature distillation is difficult. TFD, in contrast to these methods, is a completely teacher-free feature distillation that requires no additional structure. It paves the way for feature distillation design based on intermediate representations in a new direction. Some self-knowledge distillation frameworks~\cite{ref17_one,BYOT,ref19_multiple_exits,FPN} use teacher-generated methods, which construct additional auxiliary architectures (\eg, branches~\cite{ref17_one}, classifiers~\cite{ref19_multiple_exits}, FPN~\cite{FPN}) to present on-the-fly logits distillation. However, these methods are not teacher-free distillation methods. They need careful design and training of the auxiliary structures, which can make it difficult to optimize the student network~\cite{ref48-msdensenet}. These methods are teacher-based and work mainly on the outputted logits, not the intermediate features. Other Self-KDs~\cite{Self-KnowledgeDistillation,lee2020self} use multiple augmentation data for multiple networks that share the parameters of the student network. Because these methods always lose their local information during the augmentation process, they can only be used for certain tasks (\eg, object detection). In conclusion, these methods of self-knowledge distillation are not teacher-free and incur additional training costs. Compared to current self-knowledge distillation methods, our TFD, a complete teacher-free feature distillation approach, is quite different. Recent works~\cite{Instance-SpecificLabelSmoothing,Free} analyze the relationship between label smoothing~\cite{label-smoothing} and allow a student model to learn from the manually designed smooth distribution using logit KD. These methods act on the outputted logits without using teacher models, similar to label smoothing. On the other hand, TFD is the first feature distillation method without a teacher. The main experimental results show that TFD can be effectively combined with label smoothing methods to improve performance. As a result, TFD differs from these label-smoothing-based methods and expands the range of teacher-free distillations.


%Recently proposed self-knowledge distillation is primarily divided into teacher-generated based methods~\cite{ref17_one,BYOT,ref19_multiple_exits,FPN} and data augmentation based methods~\cite{Self-KnowledgeDistillation,lee2020self}. The teacher-generated based methods utilize additional auxiliary networks, which share shallow layers with student models. 


%Some self-knowledge distillation frameworks~\cite{ref17_one} use teacher-generated methods, which construct additional auxiliary architectures (\eg, branches~\cite{ref17_one}, classifiers~\cite{ref19_multiple_exits}, FPN~\cite{FPN}) to present on-the-fly logits distillation. Nevertheless, these methods are not teacher-free distillation methods. They need careful designing and training of auxiliary structures, which may make it difficult to optimize student network~\cite{ref48-msdensenet}.

%As discussed in the main paper, these methods are teacher-based and mainly work on the outputted logits, not the intermediate features. 



%The data augmentation methods use multiple augmentation data for multiple networks, which share the parameters of the student network. These methods always lose their local information in the augmentation process, limiting their application to different tasks (\eg, object detection). In summary, these self-knowledge distillation methods are not teacher-free methods and also bring additional training costs.  Our TFD, a complete teacher-free feature distillation approach, is quite different from the current self-knowledge distillation methods. 




\section{Methodology}

We begin with a general formulation in \textsection~\ref{sec: review} to review feature distillation methods in this section. Then, in\textsection~\ref{sec: Formulation}, we present the formulation and insights of our Teacher-Free Feature Distillation (TFD).

\subsection{Revisiting Conventional Feature Distillation}
\label{sec: review}
We first revisit the general formulation of feature distillation methods for a better understanding of our approach. Current feature distillation approaches~\cite{ref11_feature_kd,ref12_attention_kd} encourage the student model to mimic the intermediate features of the teacher model by explicitly optimizing the feature distillation loss. Let $x$ denote the training data and $\mathcal{Q}$ denote a set of layer location pairs for feature distillation, for a target student model $S$ with features $\bm{\phi_S}$ and its teacher $T$ with features $\bm{\phi_T}$, the general objective function can be defined as follows.

For a better understanding of our technique, we go over the general formulation of feature distillation approaches again. By explicitly optimizing the loss of feature distillation, current feature distillation approaches~\cite{ref11_feature_kd,ref12_attention_kd} encourage the student model to emulate the intermediate features of the teacher model. The generic objective function for a target student model $S$ with features $\bm{\phi_S}$ and its teacher $T$ with features $\bm{\phi_T}$ can be written as follows, using $x$ as training data and $\mathcal{G}$ as a set of layer location pairings for feature distillation.


\begin{equation}
\label{eq:kd_loss}
\mathcal{L}_{\text{S}}=\mathcal{L}_{\text{CE}}+\lambda\sum_{q\in\mathcal{G}}\mathcal{D}_{f}\big(T_s^q(\bm{\phi_S}), T_t^q(\bm{\phi_T})\big),
\end{equation}where  $\theta_S$ is the parameters of student model, respectively. $T_s$ and $T_t$ are the student and teacher transformation to align feature dimensions~(\eg, channel and spatial). $\mathcal{D}_{f}(\cdot)$  measures the difference between intermediate features. $l\ambda$ is an adjustable weighting factor that is normally initialized to a very large value and decays during training.


\subsection{Formulation of TFD}
\label{sec: Formulation}
The goal of our TFD is to achieve the distillation of characteristics using the superior features of the student model while keeping the design as general as possible. TFD achieves this goal using privileged self-features of different layers and channels, as shown in Figure~\ref{fig:main}, with only a student network and training data.


The deep layer features contain more task-relevant semantic visual concepts and obtain significant gains in the distillation framework~\cite{Cross-Layer}, according to the information-bottleneck theory~\cite{information-bottleneck,information-bottleneck2}. As a result, our depth-wise TFD supervises shallow student networks using self-features in the deep layer, which are updated by the $l_2$ loss during back propagation. The loss $\mathcal{L}_{\text{depth}}= \frac{1}{m} \sum_{i=1}^{L-1} \sum_{j>i}^{L}\mathcal{D}_{f}\big(T_{s_i}(\bm{\phi_{S_i}}), T_{s_j}(\bm{\phi_{S_j}})\big)$ of depth-wise TFD can be written as:
\begin{equation}
\begin{small}
\label{eq:inter_loss}
\mathcal{L}_{\text{depth}}= \frac{1}{m} \sum_{i=1}^{L-1} \sum_{j>i}^{L}||T_{s_i}(\bm{\phi_{S_i}}) - T_{s_j}(\bm{\phi_{S_j}})||^2,
\end{small}
\end{equation}where $m$ denotes the number of pair loss, $L$ is the number of layers of selected features, we use $l_2$ distance as ${D}_{f}$, and $T_{s}$ represents feature alignment. In particular, we use a pooling operation and channel cropping to align features in spatial and channel dimensions without complex transformation. Note that $\bm{\phi_{S_j}}$ in Equation~(\ref{eq:inter_loss}) is frozen when updating losses.



As a whole, we supervise the student network with three losses in our TFD method:
\begin{equation}
\label{eq:TfFD_loss}
\mathcal{L}_{\text{TFD}}=\mathcal{L}_{\text{CE}}+\alpha\frac{1}{m} \sum_{i=1}^{L-1} \sum_{j>i}^{L}||T_{s_i}(\bm{\phi_{S_i}}) - T_{s_j}(\bm{\phi_{S_j}})||^2,
%\vspace{-0.5em}
\end{equation}
where $\alpha$ and $\bbeta$ are the weighting factors used to scale the losses. 


\begin{table*}[t]
  \centering
   \caption{Comparison of results with advanced feature distillation methods~(\eg,  FitNets~\cite{ref11_feature_kd}, AT~\cite{ref12_attention_kd}, SP~\cite{ref41_sp}, AB~\cite{AB}, FT~\cite{ref28_factor}, NST~\cite{ref40_nst}) reported in CRD~\cite{ref14_relation_crd} in the same training setting of 240 epochs. Note that the teacher models are only for other feature distillation methods, and the TFD is completely free of teacher models. All results of other methods refer to the CRD. We report top-1 
  best accuracies (\%) for our method over 3 runs.}
   \resizebox{175mm}{!}{
    \begin{tabular}{ll|llllllll|ll|lll}
    \toprule
    Student & Top-1 & Teacher & Top-1 & FintNets & AT    & SP    & AB    & NST    & TFD & TFD$\dag$ \\
    \midrule
    ResNet20 & 69.06 & ResNet110 & 74.31 & 68.99 & 70.22 & 70.04 & 69.53 & 69.53 & 70.56 & 71.11 \\
    ResNet32 & 71.14 & ResNet110 & 74.31 & 71.06 & 72.31 & 72.69 & 70.98 & 71.96  & 72.73 & 73.22 \\
    WRN-16-2 & 73.26 & WRN-40-2 & 75.61 & 73.58 & 74.08 & 73.83 & 72.50  & 73.68 & 74.39 & 75.01 \\
    WRN-40-1 & 71.98 & WRN-40-2 & 75.61 & 72.24 & 72.77 & 72.43 & 72.38 & 72.24 & 73.08 & 74.02 \\
    \bottomrule
    \end{tabular}%
    }
  \label{tab:cifar-FD}%
  \vskip -0.1in
\end{table*}%



\begin{table*}[htbp]
  \centering
   \caption{Comparison of results with advanced distillation methods reported in CRD~\cite{ref14_relation_crd} under the same training environment of 240 epochs. These methods include logits distillation~\cite{ref10_kd}, feature distillation~(\eg, VID~\cite{ref43_vid} and PKT~\cite{ref42_pkt}) and relation distillation~(\eg, RKD~\cite{ref13_rkd}, CC~\cite{ref40_cc} and CRD~\cite{ref14_relation_crd}). Note that TFD and TFD$\dag$~(TFD+label smoothing~\cite{label-smoothing}) are completely free of teacher models. The TFD+KD and TFD+CRD employ the same teacher-based settings with KD and CRD. All results of other methods refer to CRD. We report top-1 the best accuracies (\%) for our method over 3 runs.}
   \resizebox{180mm}{!}{
    \begin{tabular}{ll|ll|llllll|llll}
    \toprule
    Student & Top-1 & Teacher & Top-1 & KD & VID & PKT & RKD & CC & CRD & TFD & TFD$\dag$ & TFD+KD & TFD+CRD \\
    \midrule
    ResNet20 & 69.06 & ResNet110 & 74.31 & 70.67 & 70.16 & 70.25 & 69.25 & 69.48 & 71.36 & 70.56 & 71.11 & 71.96 & 72.08 \\
    ResNet32 & 71.14 & ResNet110 & 74.31 & 73.08 & 72.61 & 72.61 & 71.82 & 71.48 & 73.48 & 72.73 & 73.22 & 73.88 & 73.96 \\
    WRN-16-2 & 73.26 & WRN-40-2 & 75.61 & 74.92 & 74.11 & 74.54 & 73.35 & 73.46 & 75.38 & 74.49 & 75.05 & 75.68 & 75.85 \\
    WRN-40-1 & 71.98 & WRN-40-2 & 75.61 & 73.54 & 73.30 & 73.45 & 72.22 & 72.11 & 74.04 & 73.09 & 74.02 & 74.46 & 74.76 \\
    \bottomrule
    \end{tabular}%
    }
  \label{tab:other}%
\end{table*}%






\section{Experiments}
\label{sec:Experiments}
In this section, we first evaluate our approach on CIFAR-100 in \textsection~\ref{exp:cifar100} and ImageNet in \textsection~\ref{exp:ImageNet}. We then compare the performance with existing feature distillations and regularization methods. For fair comparisons, we use the public codes of these approaches with the same training and data preprocessing settings throughout the experiments. For TFD, we set $\alpha$  and $\beta$  as 0.005 and 0.0004, respectively. Based on TFD, we apply label smoothing~\cite{goolenet} to the output logits for a fair comparison with logit-based training methods, named TFD$\dag$. Comprehensive ablation experiments are performed to analyze the key design in \textsection~\ref{exp:Ablation}. 




  
\subsection{Experiments on CIFAR-100}
\label{exp:cifar100}
%\noindent\textbf{Dataset}. CIFAR-100~\cite{cifar} is the most popular classification dataset for evaluating the performance of distillation methods. It contains 50,000 training images and 10,000 test images with 100 classes. 

\noindent\textbf{Implementation}. On CIFAR-100~\cite{cifar}, we conduct experiments on prevalent ResNets~\cite{ref20_resnet} with different depths and WRNs~\cite{wrn} with different widths. All teacher-student networks are trained in the settings of CRD~\cite{ref14_relation_crd}, whose training epochs are 240. We use a mini-batch size of 64 and a standard SGD optimizer with a weight decay of 0.0005. The multi-step learning rate is initialized to 0.05, decaying by 0.1 at 150, 180 and 210 epochs. 



\noindent\textbf{Comparison with feature distillation}. In Table \ref{tab:cifar-FD}, we compare our approach with state-of-the-art feature distillation methods under the same training settings. For ResNet20 and ResNet32, TFD obtains absolute accuracy gains $1.59\%\sim1.60\%$. Furthermore, in the WRN backbones, TFD outperforms baselines with $1.11 \%\sim1.23 \%$ margins, which shows its practical value for different depth and width networks. Compared to other feature distillation methods with a strong pre-trained teacher model, TFD achieves a performance margin $0.03\%\sim1.99\%$, indicating that our framework without the teacher model can still effectively boost performance.  


\noindent\textbf{Comparison with different types of distillations.} Apart from feature distillation, logits KD and relation distillation can also achieve state-of-the-art performance enhancements. As acting on different knowledge of distillation, our TFD could naturally combine with these methods to obtain additional gains. That is to say, the TFD is orthogonal to logits and relation distillations. Table~\ref{tab:other} introduces more comparisons and combinations of TFD with these methods.  The results of the comparison indicate that TFD can achieve considerable gains with other categories of distillation and exceeds recent advanced feature distillations~(eg, VID~\cite{ref43_vid} and PKT~\cite{ref42_pkt}) on most student models. And the experimental results prove the orthogonality of TFD and other advanced distillations~(\eg, KD~\cite{ref10_kd} and CRD~\cite{ref14_relation_crd}).



%\noindent\textbf{Comparison with feature regularization methods}. In Table~\ref{tab:cifar-FR}, we provide results of various models for baselines, TFD and other regularization methods. On different backbones, TFD obtains $0.74\%\sim1.59\%$ absolute accuracy gains and outperforms DropBlock~\cite{DropBlock} with $0.61\%\sim1.23\%$ margins. This proves that more semantic feature distortions of TFD can obtain significant performance gains.



\subsection{Experiments on ImageNet}
\label{exp:ImageNet}
%\noindent\textbf{Dataset}. We also conduct experiments on the ImageNet dataset (ILSVRC12)~\cite{imagenet}, which is considered the most challenging classification task. It contains about 1.2 million training images and 50 thousand validation images, and each image belongs to one of 1,000 categories.



\noindent\textbf{Implementation}.The ImageNet~\cite{imagenet} experiments are conducted on standard ResNet18. For ResNet18, we adopt the same training setting with most distillation methods, whose training epochs are 100. The multi-step learning rate is initialized to 0.1, decaying by 0.1 at 30, 60, and 90 epochs. 


\noindent\textbf{Comparison results}. Table~\ref{tab:imagnet-sota} reports the performance of our approach on ImageNet. TFD improves baseline models of ResNet18 by $0.8\%$ gains in top-1 accuracy. Although conventional feature distillation methods employ pre-trained ResNet34 as a teacher model, TFD still obtains very competitive performance under teacher-free settings. Compared to teacher-generated based self-knowledge distillations~(\eg, FRSKD~\cite{FPN} and BYOL~\cite{BYOT}), TFD outperforms these methods, supporting the superiority of our approach in the large-scale dataset. 

\textbf{Attention map visualization}. In our TFD, the self-feature distillation would help the network to focus on valuable information. As shown in Figure~\ref{fig:cam++}, the TFD attention map pays more attention to the important regions compared to baseline.

  \begin{figure*}[t]
      \center{\includegraphics[width=1\linewidth]{IJCNN22-822-CR/cam.pdf}}
    \caption{Comparison of the Grad-CAM++~(\cite{Grad-CAM++}) visualization results between the features of the KD, and TFD.
 }
    %\vspace{-5mm}
    \label{fig:cam++}
  \end{figure*}


\begin{table*}[t]
\caption{Top-1  accuracies (\%) on ImageNet dataset. All results of other methods are reproduced under the same training settings. TFD$\dag$ refers to TFD with Label Smoothing~\cite{goolenet} for fair comparisons with logits-based methods.}
\label{tab:imagnet-sota}
%\vskip 0.15in
\vspace{-3mm}
\begin{center}
\begin{small}
\resizebox{180mm}{!}{
    \begin{tabular}{ll|l|l|lllllll|ll}
    \toprule
    Teacher & Student & Acc & Teacher & Student & KD & ESKD & ATKD & ONE &  FRSKD & BYOT & \textbf{TFD~(ours)} &\textbf{TFD$\dag$~(ours)} \\
    \midrule
    \multirow{1}[1]{*}{ResNet-34} & \multirow{1}[1]{*}{ResNet-18} & Top-1 & 73.40 & 69.75 & 70.66 & 70.89 & 70.78 & 70.55 & 70.17 & 69.84 & \textbf{70.55} & \textbf{70.66} \\

    \bottomrule
    \end{tabular}%
}
\end{small}
\end{center}
\vspace{-2mm}
\end{table*}



\subsection{Ablation Study}
\label{exp:Ablation}
We isolate the influence of each component of our method in this section. The CIFAR-100 dataset is used in all trials. We run each setting three times and report the top-1 mean accuracies.

\noindent\textbf{Analysis of each loss}. In this section, we include more specific settings for each TFD component. Another question for depth-wise TFD is how to establish feature supervision direction. We examine three different strategies: top-down, bottom-up, and bidirectional. Knowledge of features from deep levels of a backbone network is utilized to direct the training of features from prior layers in top-down distillation. This setting is reversed in bottom-up distillation, and bi-directional distillation includes both of them.  The findings reveal that top-down and bidirectional distillation produces similar results. In the final implementation, we used top-down distillation. 

\begin{table}[htbp]
  \centering
  \caption{Different detailed settings for TFD of ResNet32 on CIFAR-100. We report top-1 mean accuracies (\%) over 3 runs.}
    \begin{tabular}{ll|lr}
    \toprule
    Loss & Setting & Top-1 & \multicolumn{1}{l}{Gain} \\
    \midrule
    \multirow{3}[2]{*} {$\mathcal{L}_{\text{TFD}}$} & Top-down & 72.61  & 1.47  \\
      & Bottom-up & 72.11  & 0.97  \\
      & Bi-directional & 72.66  & 1.52  \\
    \bottomrule
    \end{tabular}%
  \label{tab:design-loss}%
\end{table}%

%\noindent\textbf{Design of each loss}. As shown in Table~\ref{tab:Ablation-loss}, an ablation study on CIFAR100 with ResNet32 is conducted to demonstrate the individual effectiveness of different components in TFD. It is observed that (a) A single loss of TFD can also obtain $0.81\%\sim 1.20\%$ accuracy gains. (b) Compared with width-wise TFD, depth-wise loss enjoys a more noticeable boost. (c) The width-wise TFD in the deep layer obtains more obvious performance improvement. This is consistent with that some feature regularization methods~\cite{DropBlock,BeyondDF}  work well on the final stage of the neural network.

%\noindent\textbf{Advanced feature mimicking loss for TFD}. In Figure~\ref{fig:Ablation} (b), we explore different feature mimicking losses for TFD for ResNet32 on CIFAR-100. The SP~\cite{ref41_sp} and AT~\cite{ref12_attention_kd} achieve more obvious performance than the simple $l_2$ loss, indicating that the specially designed mimicking loss can further improve the performance of TFD. The NST~\cite{ref40_nst} and AB~\cite{AB} use complex feature mapping, resulting in the loss of valuable feature knowledge. We adopt the simple $l_2$ loss for TFD for all previous analyses and experiments.




%\noindent\textbf{Sensitivity study for hyper-parameters $\alpha$ and $\beta$}. The hyper-parameters $\alpha$ and $\beta$ are introduced in TFD to control the magnitude of $\mathcal{L}_{inter}$ and $\mathcal{L}_{intra}$. As shown in Figure~\ref{fig:Ablation} (c), experiments on CIFAR100 and ResNet32 are conducted to study their sensitivity. The results demonstrate that ($\alpha$, $\beta$) = (0.005, 0.0004) is the best solution for the hyper-parameter setting. Even in the worst situation when $\alpha$ = 0.05 and $\beta$ = 0.0012, TFD still achieves 0.64\% accuracy improvements than the baseline and outperforms some  distillation methods~(\eg, FitNets~\cite{ref11_feature_kd} and AB~\cite{AB}) in Table \ref{tab:cifar-FD}. These results show that our approach can achieve robust performance improvements under different hyper-parameters.



\begin{table}[htbp]
\vskip -0.15in
  \centering
  \caption{Top-1 accuracy (\%) of TFD equipped with other training techniques~(\eg,Dropout~\cite{ref01_alexnet} and Label Smoothing~\cite{goolenet},Cutout~\cite{Cutout}, AutoAugment~\cite{AutoAugment} and  Logit KD~\cite{ref10_kd} ) for ResNet32 on CIFAR-100.}
   \resizebox{43mm}{!}{
    \begin{tabular}{ll}
    \toprule
    Method & Top-1 \\
    \midrule
    Baseline & 71.14 \\
    \midrule
    TFD & 72.73 \\
    + Dropout & 72.86 \\
    + Label Smoothing & 73.20 \\
    + Cutout & 73.01 \\
    + AutoAugment& 73.17 \\
    + Logit KD& 73.88 \\
    \bottomrule
    \end{tabular}%
    }
  \label{tab:Ablation-other-training}%
\end{table}





\noindent\textbf{Augmenting TFD with other orthogonal training techniques}. TFD serves as an intermediate feature regularization, which is orthogonal to other training approaches (\eg, data augmentation and logit-based training strategies). As shown in Table~\ref{tab:Ablation-other-training} , the data augmentation methods~(\eg,Cutout~\cite{Cutout} and AutoAugment~\cite{AutoAugment}) bring $1.87\%\sim2.03\%$ accuracy gains than baseline, and label smoothing~\cite{Self-KnowledgeDistillation} obtains 2.06\% improvement. In particular, the combination of TFD with logits KD~\cite{ref10_kd} obtained 2.74\% gain under ResNet110 as a teacher model. For a fair comparison, TFD does not use other training techniques in previous experiments.


%Finally, extended experiments on object detection, more discussions and comparisons can be found in supplementary materials.

\section{Conclusion}


In this paper, we present TFD, a simple, effective, and new framework to efficiently perform feature distillation without teacher models. Based on our insight that feature distillation does not depend on additional modules, TFD achieves this goal by capitalizing on privileged self-features without defining complex transformations and assuming a teacher model to be available. TFD achieves considerable performance gains in various neural networks without adding additional training parameters or inference overhead, according to extensive tests on CIFAR-100 and ImageNet. We believe that this beautiful and practical technique will stimulate further knowledge distillation design and understanding research.



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
