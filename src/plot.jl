using Statistics
import .Plots; const pl = Plots

export plot_gp_intervals #, plot_gp_gradients, plot_gp_vector_field, plot_gp_phase_space

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

  y isa Nothing ? () : pl.scatter!(p, x_d, y, label=nothing)
  pl.plot!(p, x_d, m, ribbon = (abs.(l.-m),abs.(u.-m)), label=nothing)
  for i in 1:n_samples
    pl.plot!(p, x_d, f[:,i], color=:gray, label=nothing)
  end

  p
end
