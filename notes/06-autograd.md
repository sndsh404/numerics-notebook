# 06 Autograd

`calccode/autograd.py` is the capstone: scalar reverse-mode autograd, micrograd style. Every `Value` carries its data, a grad, and a closure that knows how to push grad back to its parents. `backward()` sorts the graph topologically, seeds the output with grad 1, and walks it in reverse applying the chain rule one local derivative at a time.

The payoff over the last two modules is exactness plus cost. Finite differences give an approximate gradient in 2n evaluations. Symbolic differentiation gives an exact gradient but the expression can explode. Reverse mode gives an exact gradient (up to float rounding) in one backward pass, and the pass costs about the same as the forward pass no matter how many inputs there are. That asymmetry is the whole reason deep learning works: one scalar loss, millions of parameters, one sweep.

On top of Value I built Neuron, Layer, and MLP. The net is [1, 10, 10, 1] with tanh hidden layers and a linear output, trained by full-batch gradient descent on mean squared error against sin(x) on [-pi, pi]. The test asserts the loss drops by at least half over 200 epochs. In practice it falls by a lot more than that, and the plot script overlays the fit on the true curve.

What surprised me: the grad accumulation rule. When one Value feeds two operations, like x in x * x, its gradient is the sum of both contributions. I knew the multivariable chain rule said this. Watching a one-line `+=` in the closure be the entire implementation of that fact was the moment the chain rule stopped being notation for me.

Where this breaks: scalars. Every weight is its own node, so a forward pass over 32 training points builds thousands of Python objects and the test takes seconds, not microseconds. Real frameworks work on tensors so one graph node covers a whole matrix, and they fuse the backward ops into kernels. The algorithm is identical to what I wrote. The engineering is everything else. Also, my grads accumulate into the parameters, so forgetting zero_grad between epochs silently averages across steps. I know because the first version of the training loop did exactly that.
