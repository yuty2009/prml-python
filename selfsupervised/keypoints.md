# Key points on self-supervised learning

## 自监督学习的一些重要观点与结论

1. **MoCo** [He, Kaiming, Chen, Xinlei, Xie, Saining, Li, Yanghao, Dollár, Piotr, and Ross Girshick. "Masked Autoencoders Are Scalable Vision Learners." ArXiv, (2021).] [[paper](https://arxiv.org/abs/2111.06377)] [[code](https://github.com/facebookresearch/mae)]
    >
    > 1. But supervised pre-training is still dominant in computer vision, where unsupervised methods generally lag behind. The reason may stem from differences in their respective signal spaces. Language tasks have discrete signal spaces (words, sub-word units, etc.) for building tokenized dictionaries, on which unsupervised learning can be based. Computer vision, in contrast, further concerns **dictionary building**, as the raw signal is in a continuous, high-dimensional space and is not structured for human communication (e.g., unlike words).
    > 2. Unsupervised learning trains encoders to perform **dictionary look-up**: an encoded “query” should be similar to its matching key and dissimilar to others.

2. **DINO** [Caron, Mathilde, Touvron, Hugo, Misra, Ishan, Jégou, Hervé, Mairal, Julien, Bojanowski, Piotr, and Armand Joulin. "Emerging Properties in Self-Supervised Vision Transformers." ArXiv, (2021).] [[paper](https://arxiv.org/abs/2104.14294)] [[code](https://github.com/facebookresearch/dino)]:
    >
    > 1. These self-supervised pre-training objectives use the words in a sentence to create pretext tasks that provide **a richer learning signal** than the supervised objective of predicting a single label per sentence. Similarly, in images, image level supervision often reduces the rich visual information contained in an image to a single concept selected from a predefined set of a few thousand categories of objects.

3. **HuBERT** [Hsu, Wei, Bolte, Benjamin, Tsai, Yao, Lakhotia, Kushal, Salakhutdinov, Ruslan, and Abdelrahman Mohamed. "HuBERT: Self-Supervised Speech Representation Learning by Masked Prediction of Hidden Units." ArXiv, (2021).] [[paper](https://arxiv.org/abs/2106.07447)] [[code](https://github.com/facebookresearch/fairseq)]
    >
    > 1. Self-supervised learning for the speech recognition domain faces unique challenges from those in CV and NLP. Firstly, the presence of multiple sounds in each input utterance breaks the **instance classification assumption** used in many CV pre-training approaches. Secondly, during pre-training, there is no prior **lexicon** of discrete sound units available, as in NLP applications in which words or word pieces are used, hindering the use of predictive losses. Lastly, the **boundaries between sound units** are not known, which complicates masked prediction pre-training.
    > 2. One crucial insight motivating this work is the importance of consistency of the targets, not just their correctness, which enables the model to focus on modeling the **sequential structure** of input data.

4. **SSL Cookbook** [Balestriero, Randall, Ibrahim, Mark, Sobal, Vlad, Morcos, Ari, Shekhar, Shashank, Goldstein, Tom, Bordes, Florian et al. "A Cookbook of Self-Supervised Learning." ArXiv, (2023).] [[paper](https://arxiv.org/abs/2304.12210)]
    >
    > 1. For example, by using different crops of a given images and positive view, the SSL model will be trained to produce a representation that is **invariant** to these different crops.
    > 2. It is worth noting that **perfect invariance** is not achieved thanks to the projector, which helps improve performance on tasks which are not entirely invariant.
    > 3. Adding a **projector** is not only useful for SSL but is also highly beneficial in a supervised training setting when there is a **misalignment** between the training and downstream tasks.
    > 4. Reducing the **misalignement** between the training and pretext task (by using class label to find the positives pair in contrastive learning) leads to learning a network for which the **best linear probe performance** on ImageNet are obtained at the **last projector layer** (instead of the backbone).
    > 5. This might imply that the projector has a role in handling inconsistent or noisy augmented views during the SSL training process.
    > 6. In fact, it is much more beneficial in SSL to increase the backbone size when training a ResNet than increasing the width or depth of the ResNet.
