---
layout: post
comments: true
title: "On using double-backpropagation on pytorch"
date: 2017-06-22
category: libraries
---

While doing some experiments that required double-backpropagation in pytorch (i.e. when you require the gradient of a gradient operation) I ran into some unexpected behavior. I found little information about it online, so I decided to write this short note.

__TL;DR__: If you need to compute the gradients through another gradient operation, you need to set the option ```create_graph=True``` on ```torch.autograd.grad```. This is described in the [Pytorch documentation](https://pytorch.org/docs/stable/autograd.html#torch.autograd.grad).

# The issue

Suppose you need to train a model with gradient descent, and the loss function (or any other part of the computational graph) requires the usage of a derivative. For instance, in Contractive Autoencoders (CAE), where the gradient of the reconstruction loss w.r.t to the weights is part of the contractive loss itself. In this case, when computing the gradient of the contractive loss w.r.t to the weights (for training the model), you need to take the second order derivative of the reconstruction loss w.r.t the weights (see the [paper](http://www.icml-2011.org/papers/455_icmlpaper.pdf)). This same problem happens in other tasks, such as meta-learning.

When implementing this in pytorch, you may use the autograd function ```torch.autograd.grad``` to compute the first-order gradients, use the result in the computation of the loss, and then backpropagate. Something along these lines:

{% highlight python %}
partial_loss = loss_function(x, y)
grad = torch.autograd.grad(partial_loss, w)[0]
total_loss = partial_loss + torch.norm(grad)

total_loss.backward()
{% endhighlight %}

Although this looks good, and it will *actually run*, it will not compute what you want: the ```total_loss.backward()``` operation will not back-propagate though the grad variable. 

# A simpler example that we can use to identify the problem

Let's create a toy example with only a few variables, that we can check the math by hand. Lets consider the following variables:

$$ a = 1 \qquad b = 2 \qquad  c = a^2 b  \qquad  d = \Big(a + \frac{\partial c}{\partial a}\Big) b  $$

Finally, let's say we need to compute $$\frac{\partial d}{\partial a}$$. We can do this analytically for this small problem:

$$ \frac{\partial c}{\partial a} = 2ab $$

$$ \frac{\partial d}{\partial a} = \frac{\partial (a + 2ab) b}{\partial a} = b(1 + 2b) = 2 (1 + 4) = 10 $$

Now, let's see what pytorch does for us:

{% highlight python %}
import torch 

a = torch.tensor(1, requires_grad=True)
b = torch.tensor(2)
c = a * a * b
dc_da = torch.autograd.grad(c, a)[0]
d = (a + dc_da) * b
dd_da = torch.autograd.grad(d, a)[0]

print('c: {}, dc_da: {}, d: {}, dd_da: {}'.format(c, dc_da, d, dd_da))
# c: 2, dc_da: 4, d: 10, dd_da: 2
{% endhighlight %}

We were expecting the result of $$\frac{\partial d}{\partial a}$$ to be 10, but pytorch computed it as 2. The reason is that by default, the torch.autograd.grad function will not create a node in the graph that can be backpropagated through. In this example, when computing $$\frac{\partial d}{\partial a}$$, pytorch effectivelly considered $$\frac{\partial c}{\partial a}$$ as a constant (with respect to a), and therefore took the gradient as  $$ \frac{\partial d}{\partial a} = \frac{\partial (a + \text{const}) b}{\partial a} = b = 2 $$.

To obtain the correct answer, we need to use the option ```create_graph=True``` on dc_da:


{% highlight python %}
import torch 

a = torch.tensor(1, requires_grad=True)
b = torch.tensor(2)
c = a * a * b
dc_da = torch.autograd.grad(c, a, create_graph=True)[0]

d = (a + dc_da) * b
dd_da = torch.autograd.grad(d, a)[0]

print('c: {}, dc_da: {}, d: {}, dd_da: {}'.format(c, dc_da, d, dd_da))
# c: 2, dc_da: 4, d: 10, dd_da: 10
{% endhighlight %}

# Conclusion

I found it a little tricky that Pytorch did not gave any errors, and simply assumed that when you compute a gradient w.r.t to a variable, you will not want to backpropagate through this node. This is counter-intuitive for me, since in all other cases, the default in pytorch *is* to backpropagate (e.g. in some iterative optimizations, you need to explicitly use ```tensor.detach()``` to avoid backpropagating through a node. I hope this note helps other people having issues with double-backpropagation in pytorch.
