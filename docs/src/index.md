# SparseGaussianProcesses.jl

┌───────────────────────────────────────┐ 
│⠀⠠⣤⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡇⠀⠀⠀⠀⠀⠀⡠⠤⢄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│ 
│⠀⠀⠀⢙⣦⣤⣄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡇⠀⠀⠀⠀⢀⠜⠀⣀⣀⠑⡄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│ 
│⠀⢀⠵⡱⠁⣧⠈⢷⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡇⠀⠀⠀⢠⣮⡶⢟⣖⡞⢍⣪⣆⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│ 
│⠀⢸⣶⠭⣈⠈⡷⡈⣧⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡇⠀⠀⢠⣾⠾⢍⢟⣿⣿⣿⣝⡿⣞⢆⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│ 
│⠀⢠⢷⢷⡾⣿⣿⣷⠺⣇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡇⠀⢀⣿⢋⣲⢟⡝⡲⠛⠻⣿⣌⣾⣞⡄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│ 
│⠀⠨⢵⣔⣻⡄⠹⣿⣗⣟⣦⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡇⠀⣾⣯⡺⣯⠿⡻⠫⢍⡒⢜⢿⣿⡽⣷⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│ 
│⠀⠈⡡⠒⠊⢹⣷⠻⡯⣟⣷⢧⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡇⣼⣿⣽⣿⡟⡰⠁⢀⡠⠜⠺⡢⣿⣿⣿⣇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│ 
│⠀⠘⠊⠉⣩⠝⠙⣽⣫⢿⣿⡎⢇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⣷⡿⣻⢻⣿⢖⠥⠚⠁⠀⠀⠀⠈⠳⣿⣿⣿⡄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│ 
│⠀⠀⡠⠎⠀⠀⠀⠘⡿⣇⢻⣽⡈⣇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⠔⠁⣿⣻⣣⡿⠉⠁⠀⠀⠀⠀⠀⠀⠀⠀⢫⣿⣿⣿⡄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│ 
│⠀⠈⠀⠀⠀⠀⠀⠀⠱⣽⣏⣿⣧⣼⡄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡰⠊⠀⣸⣿⣟⣷⠃⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⡞⣿⢾⣿⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│ 
│⠒⠒⠒⠒⠒⠒⠒⠒⠒⠚⣿⣻⣿⣷⣻⣖⠒⠒⠒⠒⠒⠒⠒⠒⠒⡲⠓⠒⢲⣿⡿⣾⠓⠒⠒⠒⠒⠒⠒⠒⠒⠒⠒⠒⠒⠚⣿⣞⣿⣷⡒⠒⠒⠒⠒⠒⠒⠒⠒⠒│ 
│⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠸⣷⣿⡿⡷⡧⣣⠀⠀⠀⠀⠀⠀⠀⢰⠁⠀⢰⣿⢷⣟⠎⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢹⢿⣿⣯⣿⠒⠒⠤⡀⠀⠀⠀⠀⠀│ 
│⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢫⣿⣿⣿⣸⣣⠱⡀⠀⠀⠀⠀⢰⠁⠀⢠⣿⢏⣟⡟⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢟⣿⣿⣿⣧⠀⠀⠈⠢⡄⠀⠀⠀│ 
│⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⡞⣿⣿⣷⣻⢆⠱⡀⠀⠀⢀⠇⠀⣠⣷⡿⡿⡝⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠘⣵⣿⣿⡻⣧⠀⠀⠀⠘⡄⠀⠀│ 
│⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠘⣿⣿⣻⣿⡿⣏⠘⡌⢈⠎⠈⣜⣿⣿⣽⡱⠁⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢹⡿⣿⢿⣻⡯⡈⠀⠀⢸⡄⠀│ 
│⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠐⣶⣶⣶⢶⣰⢆⢐⢶⡠⢂⣶⣶⣖⡖⠀⠀⡆⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢶⢲⡶⣶⣶⣆⠢⢤⢦⠄⠀│ 
│⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢹⢳⣻⣿⣟⣷⣿⣊⣱⣾⣿⡟⡞⠀⠀⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⡆⠻⡽⡿⣿⣆⣎⢎⠄⠀│ 
│⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡇⢗⠻⣿⣿⡿⣿⡻⡾⡻⡰⠁⠀⠀⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⣄⠙⢾⣿⣗⣿⢗⡆⠀│ 
│⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠘⡼⡀⠈⢺⢝⠭⠜⡱⢡⠃⠀⠀⠀⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢏⠦⡀⠀⣛⠿⣿⠇⠀│ 
│⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⢳⡒⠁⠈⠉⠉⢀⠇⠀⠀⠀⠀⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠘⡄⠈⠉⠀⢀⠏⠁⠀│ 
│⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠑⡄⠀⠀⢠⠃⠀⠀⠀⠀⠀⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠸⡀⠀⢀⡜⠀⠀⠀│ 
│⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⠒⠒⠁⠀⠀⠀⠀⠀⠀⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⠒⠁⠀⠀⠀⠀│ 
└───────────────────────────────────────┘ 

A [Flux](https://fluxml.ai)-based package for sparse Gaussian process models in [Julia](https://julialang.org).

It supports models of the form
```math
(f \mathbin{|} \boldsymbol{u})(\cdot) = (\mathcal{A}g)(\cdot) + \mathbf{K}_{(\cdot)z} (\mathbf{K}_{zz} + \mathbf\Lambda)^{-1} (\boldsymbol{u} - (\mathcal{B}g)(\boldsymbol{z}) - \boldsymbol\epsilon)
```
where ``g\sim\operatorname{GP}(0, k)``, ``\boldsymbol{u}\sim\operatorname{N}(\boldsymbol\mu, \mathbf\Sigma)``, ``\boldsymbol\epsilon\sim\operatorname{N}(\boldsymbol{0}, \mathbf\Lambda)``, and ``\mathcal{A}``, ``\mathcal{B}`` are inter-domain operators (currently only the identity operator is supported).
This [little-known formula](https://arxiv.org/abs/2002.09309) defines a Gaussian process with precisely the correct mean and variance of a standard sparse Gaussian process.
Using this approach, training is performed via doubly stochastic variational inference.

## Installation

```
pkg> add SparseGaussianProcesses
```

## Example

```julia
using Flux
using SparseGaussianProcesses

gp = SparseGaussianProcess(SquaredExponentialKernel(1))
x = reshape(-5:0.1:5, (1,:)) |> collect
y = sin.(x)

(gp,x,y) = gpu.((gp,x,y))

rand!(gp; num_samples = 16)
gp(x)

opt = ADAM()
dataset = Iterators.repeated((x,y), 1000)
cb = Flux.throttle(() -> @show(loss(gp,x,y)|>sum), 1)

Flux.train!((x,y) -> loss(gp,x,y)|>sum, Flux.params(gp), dataset, opt; cb = cb)

import Plots
plot_gp_intervals(gp, x, y)
```

## Author

- [Alexander Terenin](https://avt.im) ([PhD student](http://www.imperial.ac.uk/mathematics/), [Statistical Machine Learning](https://sml-group.cc), [Imperial College London](https://imperial.ac.uk))

## Citing

```
@article{wilson20,
	Author = {James T. Wilson and Viacheslav Borovitskiy and Alexander Terenin and Peter Mostowski and Marc Peter Deisenroth},
	Journal = {arXiv:2002.09309},
	Title = {Efficiently sampling functions from Gaussian process posteriors},
	Year = {2020}}

```