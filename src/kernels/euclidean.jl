import Flux
using Random: randn!

export SquaredExponentialKernel

"""
    EuclideanKernel

An abstract covariance kernel defined over a Euclidean space.
"""
abstract type EuclideanKernel <: CovarianceKernel end

function hyperprior_logpdf(self::EuclideanKernel)
  self.hyperprior.log_variance(self.log_variance) .+ 
    self.hyperprior.log_length_scales(self.log_length_scales)
end

Flux.trainable(k::EuclideanKernel) = (k.log_length_scales,k.log_variance)


"""
    SquaredExponentialKernel

A squared exponential kernel

``k(\\boldsymbol{x},\\boldsymbol{y}) = \\sigma^2\\exp\\left( -\\left\\lVert\\frac{\\boldsymbol{x} - \\boldsymbol{y}}{\\boldsymbol\\kappa} \\right\\rVert^2 \\right)

parameterized by log-variance ``\\ln(\\sigma^2)`` (trainable by default) and
log-length-scales ``\\ln(\\boldsymbol\\kappa)`` (trainable by default) applied 
element-wise to each dimension.
"""
struct SquaredExponentialKernel{V<:AbstractVector,H<:Hyperprior} <: EuclideanKernel
  dims              :: Tuple{Integer, Integer}
  log_variance      :: V
  log_length_scales :: V
  hyperprior        :: NamedTuple{(:log_variance, :log_length_scales), Tuple{H,H}}
end

Flux.@functor SquaredExponentialKernel

"""
    SquaredExponentialKernel(dim::Int)

Creates a squared exponential kernel of dimension `dim`.
"""
function SquaredExponentialKernel(dim::Int)
  dims = (dim, 1)
  log_variance = [0.0]
  log_length_scales = zeros(dim)
  hyperprior = (log_variance = NormalHyperprior([0.0],[1.0]), log_length_scales = NormalHyperprior(zeros(dim),ones(dim)))
  SquaredExponentialKernel(dims, log_variance, log_length_scales, hyperprior)
end

"""
    (k::SquaredExponentialKernel)(x1::AbstractMatrix, x2::AbstractMatrix)

Computes the kernel matrix for the given squared exponential kernel.
"""
function (k::SquaredExponentialKernel)(x1::AbstractMatrix, x2::AbstractMatrix)
  (_,m) = size(x1)
  (_,n) = size(x2)
  dist = pairwise_column_difference(x1,x2) ./ exp.(k.log_length_scales)
  sq_dist = dropdims(sum(dist.^2; dims=1); dims=1)
  exp.(k.log_variance) .* exp.(-sq_dist)
end


"""
    spectral_distribution(k::SquaredExponentialKernel, n::Integer = 1)

Draws `n` samples from the spectral distribution of a standard squared exponential
kernel, which is multivariate Gaussian with covariance ``2\\mathbf{I}``.
"""
function spectral_distribution(k::SquaredExponentialKernel, n::Integer = 1)
  Fl = eltype(k.log_variance)
  (id,_) = k.dims
  sqrt(Fl(2)) .* randn!(similar(k.log_variance,(id,1,n)))
end