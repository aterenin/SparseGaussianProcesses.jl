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
  d = length(k.log_length_scales)
  (_,m) = size(x1)
  (_,n) = size(x2)
  Fl = eltype(x1)
  dist_eucl = pairwise_column_difference(x1,x2)
  dist = atan.(sin.(dist_eucl),cos.(dist_eucl)) ./ exp.(k.log_length_scales)
  loop = Fl(2*pi) .* (-k.truncation_level:1:k.truncation_level) # HACK: suppress autodiff unsupported mutation error without NaNs
  loop_dist = reshape(dist, (1,d,m,n)) .+ loop
  sq_dist = dropdims(sum(loop_dist.^2; dims=2); dims=2)
  norm_const = sum(exp.(.-(reshape(loop, (1,:)) ./ exp.(k.log_length_scales)).^2); dims=(1,2))
  out = exp.(k.log_variance) .* dropdims(sum(exp.(-sq_dist); dims=1); dims=1) ./ norm_const
  m == n ? (out + out')./2 : out # symmetrize to account for roundoff error
end


"""
    spectral_distribution(k::CircularSquaredExponentialKernel, n::Integer = 1)

Draws `n` samples from the spectral distribution of a standard squared
exponential kernel on the circle.
"""
function spectral_distribution(k::CircularSquaredExponentialKernel, n::Integer = 1)
  Fl = eltype(k.log_variance)
  (id,_) = k.dims
  support = (-k.truncation_level):(k.truncation_level) |> collect
  measure = [exp(-2 * k.reference_length_scale^2 * pi^2 * n^2) for n in support] |> StatsBase.Weights
  frequency = StatsBase.sample(support, measure, n*id)
  reshape(frequency, (id,1,n))
end

function spectral_weights(k::CircularSquaredExponentialKernel, frequency::AbstractArray{<:Any,3})
  dropdims(sum(exp.(-2 .* (exp.(k.log_length_scales).^2 .- k.reference_length_scale^2) .* eltype(frequency)(pi)^2 .* frequency.^2); dims=(1,2)); dims=(1,2))
end