# [Designing Network Design Spaces](https://arxiv.org/pdf/2003.13678.pdf)

The goal of this paper is to help advance the understanding of network design and discover design principles that generalize across the setting. Instead of focusing on designing individual network instances, author design network design space that parameterize population of network.

# Introduction

In this work, author present a new network design paradigm that combines the advantages of manual design and NAS. Instead of focusing on designing individual network instances, author design design spaces that parameterize populations of network. Like in manual design, author aim for interpretability and to discover general design principles that describe network that are simple, work well, and generalize across setting. Like in NAS, author aim to take advantage of semi-automated procedures to help achieve these goal.

![Design Space Design](./DesignSpaceDesign.png)

The general strategy is to progressively design simplified version of an initial design space while maintaining or improving its quality. As shown in the image above, the design started from initial design space A and apply two refinement steps to yield design space B and C. In this case, ![C B A](https://latex.codecogs.com/svg.image?C%5Csubseteq%20B%5Csubseteq%20A), and the error distributions are strictly improving from A to B to C. The hope is that design principles that apply to model populations are more likely to be robust and generalize. The overall process is analogous to manual design, elevated to the population level and guided via distribution estimates of network design spaces.

The focus of this paper is on exploring network structure assuming standard model families including VGG, ResNet, and ResNeXt. From relatively unconstrained design space called AnyNet, author apply human-in-the-loop methodology to arrive at a low-dimensional design space consisting of simple "regular" networks called RegNet.

# Related Work

### Manual Network design

The design process behind popular networks, such as VGG, Inception, ResNet, ResNeXt, DenseNet, and MobileNet, was largely manual and focused on discovering new design choices that improve accuracy. Author share the goal of discovering new design principles. This methodology is analogous to manual design but performed at the design space level.

### Automated network design

The network design process has shifted from a manual exploration to more automated network design popularized by NAS. Unlike NAS which focuses on the search algorithm, this methodology focus on a paradigm for designing novel design spaces. Better design space an improve the efficiency of NAS search algorithm and lead to existence of better models by enriching the design space.

### Network Scaling

Both manual and semi-automated network design typically focus on finding best-performing network instances for specific regime. The goal of this paper is to discover general design principles that hold across regimes and allow for efficient tuning for the optimal network in any target regime.

### Comparing network

The author of [On network design space for visual recognition](../NDSVR) proposed a methodology for comparing and analyzing populations of networks sampled from a design space. This distribution-level view is fully alight with goal of finding general design principles. Therefore, author adopt this methodology and demonstrate that it can serve as a useful tool for the design space design process.

### Parameterization

Final quantized linear parameterization shares similarity with previous work. There are two key difference:

1. provide empirical study justifying the design choices
2. give insights tinto structural design choices that were not previously understood.

# Design Space Design

Author propose to design progressively simplified versions of an initial, unconstrained design space, refered as *design space design*. In each step of design process the input is an initial design space and the output is a refined design space, where the aim of each design step is to discover design principle that yield populations of a simpler or better performing model.

## Tools for Design Space Design

To evaluate and compare design spaces, author used [the tools introduced by Radosavovic et al](../NDSVR). The method is to quantify the quality of a design space by sampling a set of models from that design space and characterizeing the resulting model error distribution. 

To obtain a distribution of model, author sample and train n models from a design space. For efficiency author primarily do so in a low-compute, low-epoch training regime.

The primary tool for analyzing design space quality is the error empirical distribution function(EDF). The error EDF of n models with error ![e_i](https://latex.codecogs.com/svg.image?e_i) is given by:

![error empirical distribution function](https://latex.codecogs.com/svg.image?F(e)=%5Cfrac%7B1%7D%7Bn%7D%5Csum_%7Bi=1%7D%5E%7Bn%7D%7B1%5Cleft%5Be_i%3Ce%5Cright%5D%7D)

![error empirical distribution function](https://latex.codecogs.com/svg.image?F(e)) gives the fraction of model with error less than e. 

![Statistic of the AnyNetX design space](./AnyNetXDesignSpace.png)

The above image show the error EDF for n = 500 sampled models form the AnyNetX design space. Given a population for trained models, we can plot and analyze vaious network properties versus network error. Such visualization show 1D projections of a complex, high-dimensional space, and can help obtain insights into the design space. 

To summarize:

1. distribution of models obtained by sampling and training n models from a design space
2. compute and plot error EDFs to summarize design space quality
3. visualize various properties of a design space and use an empirical bootstrap to gain insight
4. use these insights to refine the design space
