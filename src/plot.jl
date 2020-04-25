using Statistics
import .Plots; const pl = Plots

export plot_gp_intervals

"""
    plot_gp_intervals(gp::GaussianProcess, x::AbstractMatrix, y::Union{AbstractMatrix,Nothing} = nothing; input_dim = 1, output_dim = 1)

Creates a simple 95% uncertainty interval plot for the GP at a set of
ordered data points ``\\boldsymbol{x}``. Optionally also plots a set of 
data points ``\\boldsymbol{y}``. For multivariateand vector GPs, 
plots the dimensions specified by `input_dim` and `output_dim`.
"""
function plot_gp_intervals(gp::GaussianProcess, x::AbstractMatrix, y::Union{AbstractMatrix,Nothing} = nothing; input_dim = 1, output_dim = 1)
  f = gp(x)[output_dim,:,:]
  n_samples = min(size(f,ndims(f)),32)

  x_d = x[input_dim,:]
  m = mean(f;dims=2) |> vec
  u = mapslices(v->quantile(v,0.975), f; dims=2) |> vec
  u_2sd = mean(f;dims=2) + mapslices(v->1.96*sqrt(var(v)), f; dims=2) |> vec
  l = mapslices(v->quantile(v,0.025), f; dims=2) |> vec
  l_2sd = mean(f;dims=2) - mapslices(v->1.96*sqrt(var(v)), f; dims=2) |> vec

  p = pl.plot(size=(800,600))

  y isa Nothing ? () : pl.scatter!(p, x_d, y[output_dim,:], label=nothing)
  pl.plot!(p, x_d, m, ribbon = (abs.(l.-m),abs.(u.-m)), label=nothing)
  for i in 1:n_samples
    pl.plot!(p, x_d, f[:,i], color=:gray, label=nothing)
  end

  p
end
