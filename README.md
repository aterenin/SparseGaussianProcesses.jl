# SparseGaussianProcesses.jl

┌───────────────────┐\
│⢰⢦⣀⣀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡆⠀⠀⢀⣴⣶⢄⠀⠀⠀⠀⠀⠀⠀⠀│\
│⢨⣿⣷⣻⣵⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡇⠀⣠⣾⣿⣿⣿⡆⠀⠀⠀⠀⠀⠀⠀│\
│⠨⡿⣿⣿⣿⣧⠀⠀⠀⠀⠀⠀⠀⠀⠀⡇⣼⣿⣿⠿⣿⣾⣿⡀⠀⠀⠀⠀⠀⠀│\
│⠸⠋⠑⢿⣿⣿⡆⠀⠀⠀⠀⠀⠀⠀⠀⣿⣿⣿⡟⠉⠺⣻⣿⣧⠀⠀⠀⠀⠀⠀│\
│⠀⠀⠀⠀⠹⣿⣿⡀⠀⠀⠀⠀⠀⠀⣼⣿⣿⠝⠀⠀⠀⠈⢿⣿⡆⠀⠀⠀⠀⠀│\
│⠉⠉⠉⠉⠉⢻⣿⣯⠉⠉⠉⠉⠉⣹⣿⣿⠏⠉⠉⠉⠉⠉⠉⣿⣿⣍⡉⠉⠉⠉│\
│⠀⠀⠀⠀⠀⠀⢿⣿⡆⠀⠀⢀⣮⣿⣿⡏⠀⠀⠀⠀⠀⠀⠀⠸⣿⣏⢟⡄⡜⠀│\
│⠀⠀⠀⠀⠀⠀⠈⢿⣿⣦⣶⣫⣿⣿⡏⡇⠀⠀⠀⠀⠀⠀⠀⠀⠹⣿⣯⣫⣦⠂│\
│⠀⠀⠀⠀⠀⠀⠀⠘⢿⣿⣿⣿⡿⡟⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠹⣿⣿⣿⡄│\
│⠀⠀⠀⠀⠀⠀⠀⠀⠀⠙⠾⠿⠏⠀⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠹⢿⠿⠀│\
│⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠱⠀│\
└───────────────────┘

[![](https://img.shields.io/badge/docs-dev-blue.svg)](https://aterenin.github.io/SparseGaussianProcesses.jl/dev)

This package implements sparse Gaussian processes models using doubly stochastic variational inference.

Unlike essentially all other Gaussian process packages, SparseGaussianProcesses.jl does *not* work with means and covariances.
Instead, it uses the [path-wise sampling technique](https://arxiv.org/abs/2002.09309) to implement ***entire function draws*** from Gaussian process posteriors, which can be evaluated deterministically at arbitrary locations once sampled.

It supports models of the form

```
(f | u)(.) = (Ag)(.) + K_{(.)z} (K_{zz} + \Lambda)^{-1} (u - (Bg)(z) - \epsilon)
```

where `g ~ GP(0, k)`, `u ~ N(\mu, \Sigma)`, `\epsilon ~ N(0, \Lambda)`, and `A`, `B` are inter-domain operators such as the identity, gradient, or convolutional patch map.
This [little-known formula](https://arxiv.org/abs/2002.09309) defines a Gaussian process with precisely the correct mean and variance of a standard sparse Gaussian process.

## Features

The following features are planned for this package.

- Evaluation of entire function draws at arbitrary locations.
- Posterior sample paths are fully differentiable, assuming a sufficiently smooth kernel.
- Strong inter-domain support, including gradient and convolutional Gaussian processes.
- Fully supports training on GPU.
- Strong support for vector-valued processes.
- Strong support for non-Euclidean domains.

## Examples

A set of examples are available in the `examples/` folder.

## Citing

```
@article{wilson20,
	Author = {James T. Wilson and Viacheslav Borovitskiy and Alexander Terenin and Peter Mostowski and Marc Peter Deisenroth},
	Journal = {arXiv:2002.09309},
	Title = {Efficiently sampling functions from Gaussian process posteriors},
	Year = {2020}}
```