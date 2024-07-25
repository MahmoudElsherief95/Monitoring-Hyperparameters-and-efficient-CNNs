# Monitoring-Hyperparameters-and-efficient-CNNs

In this assignment, the main goal is to get familiar with neural network hyperparameter search.
More specifically, you will perform hyperparameter search on some real-world data.
To prepare you for the search, we will first look at how you can monitor the training progress.

### Exercise 1: Combining Classes for Tracking 

You might not have noticed yet,a `Trainer` class was introduced.
The goal of this exercise is to extend this `Trainer` class to make use of the `Tracker`.

---------------------------------------------------------------------------------------------------------
### Exercise 2: Progress bar

Monitoring the loss early on during training can be useful
to check whether things are working as expected.
In combination with an indication of progress in training,
expectations can be properly managed early on.

---------------------------------------------------------------------------------------------------------
### Exercise 3: Tensorboard 

[Tensorboard](https://www.tensorflow.org/tensorboard)
is a library that allows to track and visualise data during and after training.
Apart from scalar metrics, tensorboard can process distributions, images and much more.
It started as a part of tensorflow, but was then made available as a standalone library.
This makes it possible to use tensorboard for visualising pytorch data.
As a matter of fact, tensorboard is readily available in pytorch.
From [`torch.utils.tensorboard`](https://pytorch.org/docs/stable/tensorboard.html),
the `SummaryWriter` class can be used to track various types of data.

---------------------------------------------------------------------------------------------------------
### Exercise 4: Always have a Backup-plan 

Apart from logging metrics like e.g. loss and accuracy,
it can often be useful to create a backup (or checkpoint) of training progress.
After all, you do not want hours of training to get lost
due to a programming error in a print statement at the end of your code.
This idea can also be useful to implement some form of early-stopping.
However, we will ignore that for now.

---------------------------------------------------------------------------------------------------------
## Hyperparameter Search

Finding good hyperparameters for a model is a general problem in machine learning (or even statistics).
However, neural networks are (in)famous for their large number of hyperparameters.
To list a few: learning rate, batch size, epochs, pre-processing, layer count, neurons for each layer,
activation function, initialisation, normalisation, layer type, skip connections, regularisation, ...
Moreover, it is often not possible to theoretically justify a particular choice for a hyperparameter.
E.g. there is no way to tell whether $N$ or $N + 1$ neurons in a layer would be better, without trying it out.
Therefore, hyperparameter search for neural networks is an especially tricky problem to solve.

###### Manual Search

The most straightforward approach to finding good hyperparameters is to just
try out *reasonable* combinations of hyperparameters and pick the best model (using e.g. the validation set).
The first problem with this approach is that it requires a gut feeling as to what *reasonable* combinations are.
Moreover, it is often unclear how different hyperparameters interact with each other,
which can make an irrelevant hyperparameter look more important than it actually is or vice versa.
Finally, manual hyperparameter search is time consuming, since it is generally not possible to automate.

###### Grid Search

Getting a feeling for combinations of hyperparameters is often much harder than for individual hyperparameters.
The idea of grid search is to get a set of *reasonable* values for each hyperparameter individually
and organise these sets in a grid that represents all possible combinations of these values.
Each combinations of hyperparameters in the grid can then be run simultaneously,
assuming that so much hardware is available, which can speed up the search significantly.

###### Random Search

Since there are plenty of hyperparameters and each hyperparameters can have multiple *reasonable* values,
it is often not feasible to try out every possible combination in the grid.
On top of that, most of the models will be thrown away anyway because only the best model is of interest,
even though they might achieve similar performance.
The idea of random search is to randomly sample configurations, rather than choosing from pre-defined choices.
This can be interpreted as setting up an infinite grid and trying only a few --- rather than all --- possibilities.
Under the assumption that there are a lot of configurations with similarly good performance as the best model,
this should provide a model that performs very good with high probability for a fraction of the compute.

###### Bayesian Optimisation

Rather than picking configurations completely at random,
it is also possible to guide the random search.
This is essentially the premise of Bayesian optimisation:
sample inputs and evaluate the objective to find which parameters are likely to give good performance.

Bayesian optimisation uses a function approximator for the objective
and what is known as an *acquisition* function.
The function approximator, or *surrogate*,
has to be able to model a distribution over function values, e.g. a Gaussian Process.
The acquisition function then uses these distributions
to find where the largest improvements can be made, e.g. using the cdf.
For a more elaborate explanation of Bayesian optimisation,
see e.g. [this tutorial](https://arxiv.org/abs/1807.02811)

This approach is less parallellisable than grid or random search,
since it uses the information from previous runs to find good sampling regions.
However, often there are more configurations to be tried out than there are computing devices
and it is still possible to sample multiple configurations at each step with Bayesian Optimisation.
Also consider [this paper](https://papers.nips.cc/paper/4522-practical-bayesian-optimization-of-machine-learning-algorithms) in this regard.

###### Neural Architecture Search

Instead of using Bayesian optimisation,
the problem of hyperparameter search can also be tackled by other optimisation algorithms.
This approach is also known as *Neural Architecture Search* (NAS).
There are different optimisation strategies that can be used for NAS,
but the most common are evolutionary algorithms and (deep) reinforcement learning.
Consider reading [this survey](http://jmlr.org/papers/v20/18-598.html)
to get an overview of how NAS can be used to construct neural networks.

## Efficient CNNs

In recent times CNNs have become more computationally efficient. Traditional convolutional layers apply filters across the entire depth of the input volume, mixing all the input channels to produce a single output channel. Depthwise separable convolutions, introduced as a key innovation in architectures like Xception, are a more efficient variant of the standard convolution operation. This process is divided into two layers: the depthwise convolution and the pointwise convolution. In the depthwise convolution, a single filter is applied per input channel, which significantly reduces the computational cost. Following this, a 1x1 convolution (pointwise convolution) is applied to combine the outputs of the depthwise layer, creating a new set of feature maps. This approach drastically reduces the number of parameters and computations, making the network more efficient and faster, which is especially beneficial for mobile and embedded devices.

<img src="https://www.researchgate.net/publication/358585116/figure/fig1/AS:1127546112487425@1645839350616/Depthwise-separable-convolutions.png" />

Squeeze-and-Excitation layers introduce an additional level of adaptivity in CNNs, enabling the network to perform dynamic channel-wise feature recalibration. Squeeze-and-Exitation blocks are usually executed after a convolutional layer or block
and before the residual connection by a series of relatively inexpensive computations

1. A three dimensional input consisting of different channels and the two spati l
dimensions is compressed into one dimension by global aver ge pooling. As a res lt
the spatial information is squeezed into one descriptor per channel.
2. The squeezed data is transformed by a two layer feed-forward neural network.  fter
the first linear layer ReLU is used as activation functi n and after the se ond a
sigmoid function is applied. This normalizes the output between 0 and 1 and can be
interpreted as the significance per channel.
3. The result is used to scale the input of the Squeeze-and-Exitation block by an element-
wise multiplication.

<img src="https://miro.medium.com/v2/resize:fit:1100/format:webp/1*bmObF5Tibc58iE9iOu327w.png" />

--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
### Exercise 5: Create an efficient CNN 

Today, neural networks frequently have millions or billions of parameters. However, CNNs have become more computationally efficient over the years. How far can you get with a limited amount of compute?

-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
### Exercise 6: Training (4 points)

In order to get a feeling for hyperparameter search, you have to try it out on some example. You can use the monitoring tools from previous exercises to log performance and get a feeling for which hyperparameters work well.





