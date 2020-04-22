using Flux; Flux.CuArrays.allowscalar(false)
using Flux: params, throttle, Optimise.@epochs
if isdefined(@__MODULE__,:LanguageServer) include("../src/SparseGaussianProcesses.jl"); using .SparseGaussianProcesses; end
using SparseGaussianProcesses
import Plots

gp = SparseGaussianProcess(SquaredExponentialKernel(1))

f(x) = 2 * sin.(x |> vec) + randn(length(x))/10
x = reshape(-5:0.1:5, (1,:)) |> collect
y = f(x)' |> collect
gp.inducing_points(reshape(-4.5:1:4.5,1,:), gp.kernel)

(gp,x,y) = gpu.((gp,x,y))

rand!(gp.prior_basis, gp.kernel)
rand!(gp; num_samples=16)
loss(gp,x,y)

opt = ADAM(0.001)
dataset = Iterators.repeated((x,y), 1000)

Flux.train!((x,y) -> loss(gp,x,y)|>sum, params(gp), dataset, opt; cb = throttle(() -> @show(loss(gp,x,y)|>sum), 1))

plot_gp_intervals(gp, x, y)
plot_gp_gradients(gp, x)