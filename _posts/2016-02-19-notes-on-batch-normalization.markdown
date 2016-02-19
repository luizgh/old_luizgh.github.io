---
layout: post
comments: true
title: "Notes on Batch Normalization"
date: 2016-02-19
category: reviews 
---

In this post, I will briefly review the paper **Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift** ([paper in arXiv](http://arxiv.org/abs/1502.03167)). This paper has been posted to arXiv on Feb-2015, and was well received and discussed in the community (with 145 citations as of Feb-2016, according to google scholar). 

# Summary of the paper
The central idea of the paper is to accelerate training by reducing the *Internal Covariate Shift* of the network. The authors argue that one of the problems for training deep neural networks is that the distribution of the layer's inputs change over time (since they are the outputs of previous layers), which they call *Internal Covariate Shift*. The objetive of Batch Normalization is to reduce this problem, in order to accelerate training. 

Previous research [1] has shown that the network converges faster when the inputs are *whitened* - that is, normalized to have zero mean, unit variance, and decorrelated (diagonal covariance). This paper brings this idea to the other layer's inputs (i.e. outputs from previous layers in the network). The solution proposed by the authors is to normalize the units of a layer, before the activation, using the statistics from mini-batches of data. In practice, let's consider an unit $$x$$, a neuron in the the pre-activation output of a layer in the network. We first calculate a normalized version of this unit:

$$\hat{x} = \frac{x - \text{E}[x]}{\sqrt{\text{Var}[x]}}$$

Where the expectation and variance are calculated in a mini-batch of examples. Note that this formulation only normalizes the mean to zero and the variance to one, but do not de-correlate the units within the layer. The authors argue that calculating the covariance matrix for the units would not be pratical given the size of the mini-batches vs. the number of units in each layer. 

One problem with simply normalizing the units is that it can change the representation power of the network (e.g. in the case of a sigmoid, this can lead the units to be only in the (near) linear part of the activation). In order not to lose representation power of the network, the authors introduce other two parameters (per neuron in the network): $$\gamma$$, $$\beta$$, that can "undo" this normalization:

$$y = \gamma \hat{x} + \beta$$

With these two parameters, the output has the same representation power, and the network can undo the normalization, if this is the optimal thing to do.

Besides the theoretical arguments for using this strategy, the authors report several advantages found during experiments, conducted using the ImageNet dataset, with modified versions of the Inception[2] model. Most notably, using Batch Normalization (BN), they were able to use a **30x larger learning rate** for training, obtaining the same level of performance with **14 times fewer training steps**. The authors also noted that using BN reduced the need for Dropout, showing that it can help regularize the network. Lastly, the authors used an ensemble of 6 models trained with BN to achieve the state-of-the-art results on the ImageNet LSVRC challenge (4.82% error in the test set).

# Comments and opinions

I found it very surprising that the benefits for using Batch Normalization are so large, for such a simple idea. In general, the paper is well written, the claims are well founded and the ideas are easy fo follow.

One thing I liked about this approach, besides speeding up training, is that it makes the network much more stable to the initial values assigned to the weights. In particular, the authors show that the back-propagation through a layer with Batch Normalization is invariant to the scale of its parameters.
This means that using Batch Normalization requires less time tweaking the initial parameter values.
I have recently implemented Batch Normalization for training a 7-layer CNN on the CIFAR-10 dataset. Initializing the network with small random weights (e.g. $$W \sim \text{U}(-a,a)$$ for $$a = 0.001$$), without BN the network did not train at all. I noticed that with this (poor) initialization, the pre-activation outputs decreased for each layer in the network reaching the order of $$10^{-11}$$ in the last layer, compromising training. Surprisingly, just adding BN on the last layer (right before applying softmax) was enough for the network to train properly, even with a bad initialization scheme. 

There is one thing in particular that remained unclear to me after reading the article: why Batch Normalization helps regularizing the network. The authors simply state that "a training example is seen in conjunction with other examples, and the training network no longer producing deterministic values for a given training example". In my opinion, this does not seem to properly explain why it is so good in regularizing the network (to the point of not requiring dropout). It seems that there is still a lot of room to explore and understand this idea in future research.


# References

[1] LeCun, Y., Bottou, L., Orr, G., and Muller, K. Efficient backprop. In Orr, G. and K., Muller (eds.), Neural Networks: Tricks of the trade. Springer, 1998b.

[2] Szegedy, Christian, et al. "Going deeper with convolutions." Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2015.
