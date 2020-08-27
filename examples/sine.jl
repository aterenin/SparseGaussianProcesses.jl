using Flux; Flux.CuArrays.allowscalar(false)
using Flux: params, throttle, Optimise.@epochs
using SparseGaussianProcesses
import Plots

gp = SparseGaussianProcess(SquaredExponentialKernel(1), inducing_points = PseudoDataInducingPoints)


f(x) = 2 * sin.(x |> vec) + randn(length(x))/10
x = reshape(-5:0.1:5, (1,:)) |> collect
y = f(x)' |> collect
gp.inducing_points(reshape(-5:1:5,1,:), gp.kernel)

(gp,x,y) = gpu.((gp,x,y))

rand!(gp.prior_basis, gp.kernel; num_features=67)
rand!(gp; num_samples=17)
loss(gp,x,y)|>sum

opt = ADAM(0.01)
dataset = Iterators.repeated((x,y), 200)

Flux.train!((x,y) -> loss(gp,x,y)|>sum, params(gp), dataset, opt; cb = throttle(() -> @show(loss(gp,x,y)|>sum), 0.1))

(gp,x,y) = cpu.((gp,x,y))

plot_gp_intervals(gp, x, y)