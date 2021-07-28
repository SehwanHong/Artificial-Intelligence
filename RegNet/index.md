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

