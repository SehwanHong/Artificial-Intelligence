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
2. give insights into structural design choices that were not previously understood.

# Design Space Design

Author propose to design progressively simplified versions of an initial, unconstrained design space, referred as *design space design*. In each step of design process the input is an initial design space and the output is a refined design space, where the aim of each design step is to discover design principle that yield populations of a simpler or better performing model.

## Tools for Design Space Design

To evaluate and compare design spaces, author used [the tools introduced by Radosavovic et al](../NDSVR). The method is to quantify the quality of a design space by sampling a set of models from that design space and characterizing the resulting model error distribution. 

To obtain a distribution of model, author sample and train n models from a design space. For efficiency author primarily do so in a low-compute, low-epoch training regime.

The primary tool for analyzing design space quality is the error empirical distribution function(EDF). The error EDF of n models with error ![e_i](https://latex.codecogs.com/svg.image?e_i) is given by:

![error empirical distribution function](https://latex.codecogs.com/svg.image?F(e)=%5Cfrac%7B1%7D%7Bn%7D%5Csum_%7Bi=1%7D%5E%7Bn%7D%7B1%5Cleft%5Be_i%3Ce%5Cright%5D%7D)

![error empirical distribution function](https://latex.codecogs.com/svg.image?F(e)) gives the fraction of model with error less than e. 

![Statistic of the AnyNetX design space](./AnyNetXDesignSpace.png)

The above image show the error EDF for n = 500 sampled models form the AnyNetX design space. Given a population for trained models, we can plot and analyze various network properties versus network error. Such visualization show 1D projections of a complex, high-dimensional space, and can help obtain insights into the design space. 

To summarize:

1. distribution of models obtained by sampling and training n models from a design space
2. compute and plot error EDFs to summarize design space quality
3. visualize various properties of a design space and use an empirical bootstrap to gain insight
4. use these insights to refine the design space

# The AnyNet Design Space

![general network structural for models in AnyNet design space](./AnyNetDesignSpace.png)

The basic design of networks in AnyNet design space is straightforward, as shown in the image above. Given an input image, a network consist of a simple stem, followed by the network body that performs the bulk of the computation, and a final network head that predicts the output class.

The network body consists of 4 stages operating at progressively reduced resolution. Each stage consists of a sequence of identical blocks. In total, for each stage i,  the degrees of freedom include the number of blocks, block width, and any other block parameters.

![The X block](./Xblock.png)

Most of the experiments in this paper use the standard residual blottlenecks block with group convolution called X block. As represented in the image above, each X block consist of a ![](https://latex.codecogs.com/svg.image?1\times1) conv, ![](https://latex.codecogs.com/svg.image?3\times3) group conv and a final ![](https://latex.codecogs.com/svg.image?1\times1) conv. ![](https://latex.codecogs.com/svg.image?1\times1) convs alters the channel width. Batch normalization and ReLU follow each convolution. The block has 3 parameters: the width ![w_i](https://latex.codecogs.com/svg.image?w_i), bottleneck ratio ![b_i](https://latex.codecogs.com/svg.image?b_i), and group width ![g_i](https://latex.codecogs.com/svg.image?g_i). 

The AnyNet built with this sturcture is called AnyNetX. In this design space, there are 16 degrees of freedom as each network consist of 4 stages and each stage i has 4 parameters:

1. the number of blocks ![d_i](https://latex.codecogs.com/svg.image?d_i)
2. block width ![w_i](https://latex.codecogs.com/svg.image?w_i)
3. bottleneck ratio ![b_i](https://latex.codecogs.com/svg.image?b_i)
4. group width ![g_i](https://latex.codecogs.com/svg.image?g_i)

To obtain valid models, author perform log-uniform samping of ![number of block](https://latex.codecogs.com/svg.image?d_i%5Cleq%2016), ![block width](https://latex.codecogs.com/svg.image?w_i%5Cleq%20128),  and divisible by 8, ![bottleneck ratio](https://latex.codecogs.com/svg.image?b_i%5Cin%5Cleft%5C%7B1,2,4%5Cright%5C%7D), and ![group width](https://latex.codecogs.com/svg.image?g_i%5Cin%5Cleft%5C%7B1,2,4,%5Ccdots,32%5Cright%5C%7D). From these parameters, repeat sampling until n=500, and train each model for 10 epochs.

![Statistic of the AnyNetX design space](./AnyNetXDesignSpace.png)

Basic statistics for AnyNetX is shown in the above image.

From above parameters, there are ![approximation of possible models](https://latex.codecogs.com/svg.image?(16%5Ccdot128%5Ccdot3%5Ccdot6)%5E4%5Capprox10%5E%7B18%7D) possible model configurations in the AnyNetX design space. Rather than searching the best model from over ![](https://latex.codecogs.com/svg.image?10%5E%7B18%7D) configurations, author explore to find general design principles that explains and refine the design space.

There are four purpose in approach of designing design space:
1. to simplify the structure of the design space
2. to improve the interpretability of the design space
3. to improve or maintain the design space quality
4. to maintain model diversity in the design space

### AnyNetXA

Initial unconstrained AnyNetX design space is AnyNetXA

### AnyNetXB

Shared bottleneck ratio ![Shared bottleneck ratio](https://latex.codecogs.com/svg.image?b_i=b) for all stage i for the AnyNetXA design space is called AnyNetXB. Same as AnyNetXA, author sampled and trained 500 models from AnyNetXB.

![AnyNetXA and AnyNetXB](./AnyNetXAB.png)

The EDFs of AnyNetXA and AnyNetXB, shown in the image above, are virtually identical in both in the average and best case. Therefore, this indicates when coupling the bottleneck ratio does not effect the accuracy. In addition to being simpler, the AnyNetXB is more amenable to analysis.

### AnyNetXC

![AnyNetXB and AnyNetXC](./AnyNetXBC.png)

The second refinement step closely follows the first. AnyNetXC use a shared group width ![Shared Group width](https://latex.codecogs.com/svg.image?g_i=g) over AnyNetXB. Overall, AnyNetXC has 6 fewer degrees of freedom than AnyNetXA, and reduce the design space size nearly four orders of magnitude. 

### AnyNetXD

![Example good and bad AnyNetXC networks](./GoodNBadAnyNetXC.png)

Examining the network structures of both good and bad network from AnyNetXC in image above. Top three graph represent good AnyNetXC Networks and bottom three represnet bad AnyNetXC.

From these graphs, there is a pattern: good networks have increasing widths. Applying these desing principle of ![increasing width](https://latex.codecogs.com/svg.image?w_%7Bi&plus;1%7D%5Cgeq%20w_i) to AnyNetXC and refer to the design space as AnyNetXD. 

![AnyNetXC and AnyNetXD](./AnyNetXCD.png)

The graph above represent testing different constraints on the width of the network. When using increasing widths, AnyNetXD, EDF is improved substantially. 

### AnyNetXE

There is another interesting trend. The stage depth ![d_i](https://latex.codecogs.com/svg.image?d_i) tends toincrease for the best models, although not necessarily in the last stage.

![AnyNetXD and AnyNetXE](./AnyNetXDE.png)

Applying constraints ![increasing depth](https://latex.codecogs.com/svg.image?d_%7Bi&plus;1%7D%5Cgeq%20d_i) on AnyNetXD is called AnyNetXE. AnyNetXE is slightly better than AnyNetXD.

The constraints on ![d_i](https://latex.codecogs.com/svg.image?d_i) and ![w_i](https://latex.codecogs.com/svg.image?w_i) each reduce the design space by 4!, with a cumulative reduction of ![](https://latex.codecogs.com/svg.image?O(10%5E7)) from AnyNetXA.

