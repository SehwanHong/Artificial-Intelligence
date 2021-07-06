# [MobileNetV2: Inverted Residuals and Linear Bottlenecks](https://arxiv.org/pdf/1801.04381.pdf)

In this paper, author describes a new mobile architecture, MobileNetV2, that improves the performance of mobile models on multiple tasks and benchmarks.

# Introduction

Neural networks have revolutionized many areas of machine intelligences, enabling superhuman accuracy for challenging image recognition tasks. The drive to improve accuracy often comes at a cost: modern state of art networks require high computational resources beyond the capabilities of many mobile and embedded applications.

This paper introduces a new neural network architecture that is specifically tailored for mobile and resource constrained environments. To retain the same accuracy while decreasing the number of operations and memory usage, author introcues a novel layer module: inverted residual with linear bottleneck.

# Related Work

# Preliminaries, discussion and intuition

## Depthwise separable convolution

## Linear Bottlenecks

## Inverted residuals

## Information flow interpretation

# Model Architecture

# Implementation Notes

## Memory efficient inference

# Experiments

## ImageNet Classification

## Object Detection

## Semantic Segmentation

## Ablation Study

## [Link to Neural Net](../../)
## [Link to MobileNet](../)