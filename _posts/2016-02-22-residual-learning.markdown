---
layout: post
comments: true
title: "Deep Residual Learning for Image Recognition"
date: 2016-02-22
category: reviews 
---


In this post I review the artice **Deep Residual Learning for Image Recognition** ([link to arXiv](http://arxiv.org/abs/1512.03385)) that won the 1st place in the ILSVRC 2015 classification task (ImageNet)[1] with 3.57% top-5 error, using a CNN containing **152 layers** (!), but with a reasonable computational time (less than previous models, such as VGG-19, that contained 19 layers[2]). This architecture was also used to achieve 1st place in the ImageNet detection and localization, as well as 1st place in the COCO 2015 competitions [3] (in detection and segmentation). That's quite an impressive feat.


# Summary of the paper

The central idea of this paper is to explore network depth, and in particular, how to be able to train networks with hundreds of layers. Experimental results (from previous research) demonstrate the benefits of depth in Convolutional Neural Networks. However, training deeper networks present some challenges. One problem is the vanishing/exploding gradients - which has been handled to a large extent with Batch Normalization [4]. However, even with gradients properly flowing to the first layers, naively training deeper and deeper networks does not usually increase performance. As the authors note, **as we increase depth, accuracy saturates, and then degrades rapidly**. Most surprisingly, not only the testing error gets worse, but the training error as well. However, consider the following insight: if we train a network with **L** layers, we could have a network with **L + n** layers, where the last n layers are the identity mapping. Clearly, this network has the same error as the the one with L layers, and optimizing it should get us a lower (or equal) training error. However, training a network with **L + n** layers from scratch often gives worse training performance than a network with **L** layers, showing that some networks are harder to optimize. 

With this insight, the authors propose to address this "degradation problem" by letting the layers learn a **residual mapping**. That is, instead of the layers learning a transformation $$\mathcal{H}(\textbf{x})$$, they consider that this transformation breaks down as the input plus a residual: $$\mathcal{H}(\textbf{x}) = \textbf{x} + \mathcal{F}(\textbf{x}) $$, and learn $$\mathcal{F}(\textbf{x})$$ only. The authors argue that, in the extreme case, where the identity mapping is optimal, it is easier for the layers to learn $$\mathcal{F}(\textbf{x}) = \textbf{0}$$  than to learn an identity tranformation (from a stack of non-linear layers).

In their experiments, the authors consider networks with 18, 34 , 50, 101 and 152 layers, trained on the ImageNet dataset. This residual computation is added after blocks of 2 or 3 layers (meaning that each 2 or 3 layers compute the residual of the transformation for the next level).  Their results show that this strategy is quite effective in making the learning problem easier to optimize: without adding this residual architecture, the network with 34 layers perform worse (even in the training set) than the 18-layer network. With their strategy, increasing the number of layers continue to improve performance, not only in the training set, but on the testing set as well. Surprisingly, the 152-layer network achieves 4.49% top-5 error, which is **lower than the results from ensembles of models used in previous years**. The authors also performed experiments on the smaller dataset CIFAR-10, with tens of layers, and an extreme case of 1202 layers. With 110 layers, they achieved 6.43% error (state of the art for this dataset). More surprisingly, the model with over a thousand layers converged, having a train error lower than 0.1% (although it overfit the testing set, achieving 7.93% of error on the test set).


# Comments and opinions {#comments}

Similarly to Batch Normalization, I found impressive that the idea for this paper is quite simple, yet it performs incredibly well in practice (for instance, I found this model much simpler than the ILSVRC 2014 winner - GoogLeNet, with the Inception modules). 

On the other hand, I found it hard to interpret what the model is doing. When I think about what a network with multiple layers is computing, I picture each layer projecting the input to a different feature space, slowly disentangling the inputs, so that they can be linearly classified in the last layer (e.g. I imagine something like [this](http://colah.github.io/posts/2014-03-NN-Manifolds-Topology/)). However, in this formulation, we explicitly add back the input when computing the output of the layer (or, the stack of layers to be more precise): $$y = \textbf{x} + \mathcal{F}(\textbf{x})$$. Intuitively, this seems to pull the output of the layer closer to the original feature space, which would make it harder to disentangle the factors of variation in the input. It would be interesting to generate visualizations for this network (e.g. [of this kind](http://yosinski.com/deepvis)) to help understand what the network is learning.



# References

[1] Russakovsky, Olga, et al. "Imagenet large scale visual recognition challenge." International Journal of Computer Vision 115.3 (2015): 211-252.

[2] Simonyan, Karen, and Andrew Zisserman. "Very deep convolutional networks for large-scale image recognition." arXiv preprint arXiv:1409.1556 (2014).

[3] Lin, Tsung-Yi, et al. "Microsoft coco: Common objects in context." Computer Vision–ECCV 2014. Springer International Publishing, 2014. 740-755.

[4] Ioffe, Sergey, and Christian Szegedy. "Batch normalization: Accelerating deep network training by reducing internal covariate shift." arXiv preprint arXiv:1502.03167 (2015).
