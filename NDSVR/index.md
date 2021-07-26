# [On Network Design Spaces for Visual Recognition](https://arxiv.org/pdf/1905.13214.pdf)

In this paper, author introduce new comparison paradigm of distribution estimates. Compared to comparing point and curve estimates of model familes, distribution estimates paint a more complete picture of entire design landscape

# Introduction

There has been a general trend toward better empiricism in the literature on network architecture.

![Comparing Network using different estimations](./ComparingNetworks.png)

In the early development stages, the simple method was used. The progress of the neural network was measured by simple point estimates. The architecture was makred superiror if it achieved lower error on a benchmark dataset, often irrespective of model complexity.

The improved methodoloy adopted in more recent work compares curve estimates.  This methods explore design tradeoffs of network architectures by instantiating a handful of models from a loosely defined model familes and tracing curve of error vs. model complexity. In this estimation, one model family is considered superior if it acheives lower error at every point along a curve. In this example, ResNeXt are considered better than ResNet because, ResNeXt have lower error in all the point.

However, there is some draw back in using this methodology. Curve estimates does not consider confounding factors, which may vary between model familes and may be suboptimal for one of them.

Rather than varying a single network hyperparameter while keeping all others fixed, what if instead we varu all relevant network hyperparameters? However, this is not feasible, because there are often infinite number of possible models. Therefore, author introduce a new comparison paradigm: that of distribution estimates.

Unlike Curve estimates where  they compare a few selected members of a model family, distribution estimates sample modles from a design space, parameterizing possible architectures, giving rise to distrivutions of error rates and model complexities.

This methodology rocuses on characterizing the model family. Thus enable research into designing the design space for model search.

# Related Work

### Reproducible resarch

There has been an encouraging recent trend toward better reproducibility in machine learning. Thus author share the goal of introducing a more robust methodology for evaluating model architectures in the domain of visual recognition

### Empirical studies

In the absence of rigorous theoretical understanding of deep networks, it is imperative to perform large-scale studies of deep networks to aid development. Empricial studies and robust methodology play in enabling progress toward discovering better architectures.

### Hyperparameter search

General hyperparameter search techniques address the laborious model tuning process in machine learning. In this work, author propose to directly compare full model distributions not just their minima.

### Neural architecture Search

NAS has proven effective for learning networks architecctures. A NAS instantiation has two components: a network design space and a search algorithm. Most work on NAS focuses on the search algorithm. However, in this work, author focus on characterizing the model design space.

### Complexity measures

In this work, author focus on analyzing network design space while controlling for confounding factors like network complexity. Author adopt commonly-used network complexity measures, including number of model parameters or multiply-add operations. 

# Design Spces

## Definitions

### Model family

A model family is large (possibly infinite) collection of related neural network architectures, typically shraing some high-level architectural structures or design principles (residual connections)

### Design Space

Performing empirical studies on model families are difficult since they are broadly defined and typically not fully specified. To make disctinctino between abstract model families, design space is introduces. Design space is a concrete set of architectures taht can be isntantiated from the model family.

A design space consist of two components
1. parameterization of a model family
2. a set of allowable values for each hyper parameters.

### Model distribution

As design space an contain an eponential number of candidate models, exhaustive evaluation is not feasible. Therefore, from a design space, author sampled and excaluated a fixed set of models, giving rise to a model distribution. Then, author use classical statistics for analysis. Any standard distribution, as well as learned distributions like in NAS, can be integrated into our paradigm.

### Data Generation

To analyze network design spaces, author smaple and evaluate numerous models from each design space. In doing so, author generate datasets of trained models upon which we perform empirical studies.

## Instantiations

### Model family

Author study three standarad model families:
1. Vanilla model family (feedforward network loosely inspired by VGG)
2. ResNet model family
3. ResNeXt model family

### Design space

![Design Space Parameterization](./DesignSpaceParameterization.png)

As shown in the table above, author used networks consisting of a stem, followed by three stages and a head, as described in the table above.

* For ResNet design space, a single block consists of two convolutions and residual connection.
* Vanilla design space uses an identical block structure as ResNet but without residual connection.
* In the Case of ResNeXt design spaces, we use bottleneck blocks with groups.

![Design Spaces](./DesignSpace.png)

This table above specify the set of allowable hyperparameters for each. The notation, ![a, b, n](https://latex.codecogs.com/svg.image?a,b,c) means we sample, n values sapced about evenly in log spaces in the range a to b. From these value, we select independently for three network stages, number of blocks ![d_i](https://latex.codecogs.com/svg.image?d_i), and the number of channels per block ![w_i](https://latex.codecogs.com/svg.image?w_i).

From these number, the total number of models  is ![number of models without groups](https://latex.codecogs.com/svg.image?(dw)^3) for models without groups and ![number of models with groups](https://latex.codecogs.com/svg.image?(dwrg)^3) for models with groups.

### Model distribution

Author generate model distributions by uniformly sampling hyperparameters from the allowable values for each design spaces.

### Data generation

Author uses image classification models trained on CIFAR-10. This setting enables large-scale analysis and is used often used as a testbed for recognition networks. From the above table, author selected 25k models and used to evaluate the methodology


