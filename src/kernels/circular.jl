import Flux
import StatsBase
using Random: randn!

export CircularSquaredExponentialKernel

"""
    CircularKernel

An abstract covariance kernel defined over a circle, or more generally a 
``d``-dimensional torus.
"""
abstract type CircularKernel <: EuclideanKernel end

function hyperprior_logpdf(self::CircularKernel)
  self.hyperprior.log_variance(self.log_variance) .+ 
    self.hyperprior.log_length_scales(self.log_length_scales)
end

Flux.trainable(k::CircularKernel) = (k.log_length_scales,k.log_variance)


"""
    CircularSquaredExponentialKernel

A squared exponential kernel on the circle, defined on the circle or, if 
``d > 1``, the ``d``-dimensional torus. The expression is given by

``k(x,x') = \\sum_{n \\in 2\\pi\\mathbb{Z}^d} \\sigma^2\\exp\\left( 
           -\\left\\lVert\\frac{x - x' + n}{\\kappa} \\right\\rVert^2 \\right)``

parameterized by log-variance ``\\ln(\\sigma^2)`` (trainable by default) and
log-length-scales ``\\ln(\\kappa)`` (trainable by default) applied 
element-wise to each dimension. The sum is computed to a specified truncation
level.
"""
mutable struct CircularSquaredExponentialKernel{V<:AbstractVector,H<:Hyperprior} <: CircularKernel
  dims              :: Tuple{Integer, Integer}
  log_variance      :: V
  log_length_scales :: V
  truncation_level  :: Int
  reference_length_scale :: Rational{Int}
  hyperprior        :: NamedTuple{(:log_variance, :log_length_scales), Tuple{H,H}}
end

Flux.@functor CircularSquaredExponentialKernel

"""
    CircularSquaredExponentialKernel(dim::Int)

Creates a squared exponential kernel defined on the circle, or more generally a 
torus of dimension `dim`.
"""
function CircularSquaredExponentialKernel(dim::Int)
  if dim!=1 error("Not implemented") end
  dims = (dim, 1)
  log_variance = [0.0]
  log_length_scales = zeros(dim)
  truncation_level = 10
  reference_length_scale = 1//10
  hyperprior = (log_variance = NormalHyperprior([0.0],[1.0]), log_length_scales = NormalHyperprior(zeros(dim),ones(dim)))
  CircularSquaredExponentialKernel(dims, log_variance, log_length_scales, truncation_level, reference_length_scale, hyperprior)
end

"""
    (k::CircularSquaredExponentialKernel)(x1::AbstractMatrix, 
                                          x2::AbstractMatrix)

Computes the kernel matrix for the given circular squared exponential kernel.
"""
function (k::CircularSquaredExponentialKernel)(x1::AbstractMatrix, x2::AbstractMatrix)
  (_,m) = size(x1)
  (_,n) = size(x2)
  (d,_) = k.dims
  Fl = eltype(x1)
  
  loop = Fl(2*pi) .* (-k.truncation_level:1:k.truncation_level) # HACK: only correct for d=1!
  dist = (reshape(pairwise_column_difference(x1,x2), (1,d,m,n)) .+ loop) ./ reshape(exp.(k.log_length_scales), (1,d,1,1))
  sq_dist = dropdims(sum(dist.^2; dims=2); dims=2)
  kernel = exp.(k.log_variance) .* dropdims(sum(exp.(.-sq_dist); dims=1); dims=1)

  m == n ? (kernel + kernel')./2 : kernel # symmetrize to account for roundoff error
end

"""
    spectral_distribution(k::CircularSquaredExponentialKernel, 
                          num_samples::Integer)

Draws `n` samples from the spectral distribution of a standard squared
exponential kernel on the circle.
"""
function spectral_distribution(k::CircularSquaredExponentialKernel; num_samples::Integer)
  (id,_) = k.dims
  support = (-k.truncation_level):(k.truncation_level) |> collect
  measure = [exp(-1//4 * k.reference_length_scale^2 * n^2) for n in support] |> StatsBase.Weights
  frequency = StatsBase.sample(support, measure, num_samples*id)
  reshape(frequency, (id,1,num_samples))
end

function spectral_weights(k::CircularSquaredExponentialKernel, frequency::AbstractArray{<:Any,3})
  Fl = eltype(frequency)
  loop = -k.truncation_level:1:k.truncation_level # HACK: only correct for d=1!
  norm_const_ls = sum(exp.(-1//4 .* reshape(loop, (1,:)).^2 .* exp.(2 .* k.log_length_scales)); dims=(1,2))
  norm_const_std = sum(exp.(-1//4 .* loop.^2 .* k.reference_length_scale.^2); dims=1)
  ratio = dropdims(sum(exp.(-1//8 .* (exp.(2 .* k.log_length_scales) .- k.reference_length_scale.^2) .* frequency.^2); dims=(1,2)); dims=(1,2))
  (exp.(k.log_variance ./ 2) .* ratio .* sqrt.(norm_const_std ./ norm_const_ls), 1)
end