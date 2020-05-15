import Flux
using Random: randn!
using TensorCast

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

``k(x,x') = \\sigma^2\\exp\\left( -\\left\\lVert\\frac{x - x'}{\\kappa} 
                                                     \\right\\rVert^2 \\right)``

parameterized by log-variance ``\\ln(\\sigma^2)`` (trainable by default) and
log-length-scales ``\\ln(\\kappa)`` (trainable by default) applied 
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

Draws `n` samples from the spectral distribution of a standard squared 
exponential kernel, which is multivariate Gaussian with covariance ``2 I``.
"""
function spectral_distribution(k::SquaredExponentialKernel, n::Integer = 1)
  Fl = eltype(k.log_variance)
  (id,_) = k.dims
  sqrt(Fl(2)) .* randn!(similar(k.log_variance,(id,1,n)))
end


"""
    (k::LeftGradientKernel{<:SquaredExponentialKernel})(x1::AbstractMatrix, 
                                                        x2::AbstractMatrix)

Computes the kernel matrix for the given gradient squared exponential 
cross-covariance.
"""
function (k::LeftGradientKernel{<:SquaredExponentialKernel})(x1::AbstractMatrix, x2::AbstractMatrix)
  (_,m) = size(x1)
  (_,n) = size(x2)
  (d,_) = k.parent.dims

  dist = pairwise_column_difference(x1, x2) ./ exp.(k.parent.log_length_scales)
  sq_dist = dropdims(sum(dist.^2; dims=1); dims=1)
  kernel = exp.(k.parent.log_variance) .* exp.(-sq_dist)

  dist_sc = dist ./ exp.(k.parent.log_length_scales)
  @cast out[D,M,N] := -2 * kernel[M,N] * dist_sc[D,M,N]
  reshape(out, (d*m,n))
end


"""
    (k::RightGradientKernel{<:SquaredExponentialKernel})(x1::AbstractMatrix, 
                                                         x2::AbstractMatrix)

Computes the kernel matrix for the given gradient squared exponential 
cross-covariance.
"""
function (k::RightGradientKernel{<:SquaredExponentialKernel})(x1::AbstractMatrix, x2::AbstractMatrix)
  (_,m) = size(x1)
  (_,n) = size(x2)
  (d,_) = k.parent.dims

  dist = pairwise_column_difference(x1, x2) ./ exp.(k.parent.log_length_scales)
  sq_dist = dropdims(sum(dist.^2; dims=1); dims=1)
  kernel = exp.(k.parent.log_variance) .* exp.(-sq_dist)

  dist_sc = dist ./ exp.(k.parent.log_length_scales)
  @cast out[M,D,N] := 2 * kernel[M,N] * dist_sc[D,M,N]
  reshape(out, (m,d*n))
end


"""
    (k::GradientKernel{<:SquaredExponentialKernel})(x1::AbstractMatrix, 
                                                    x2::AbstractMatrix)

Computes the kernel matrix for the given gradient squared exponential kernel.
"""
function (k::GradientKernel{<:SquaredExponentialKernel})(x1::AbstractMatrix, x2::AbstractMatrix)
  (_,m) = size(x1)
  (_,n) = size(x2)
  (d,_) = k.parent.dims

  dist = pairwise_column_difference(x1, x2) ./ exp.(k.parent.log_length_scales)
  sq_dist = dropdims(sum(dist.^2; dims=1); dims=1)
  kernel = exp.(k.parent.log_variance) .* exp.(-sq_dist)

  dist_sc = dist ./ exp.(k.parent.log_length_scales)
  scale_matrix = Diagonal(exp.(k.parent.log_length_scales))
  @cast out[D1,M,D2,N] := (2*scale_matrix[D1,D2] - 4*dist_sc[D1,M,N] * dist_sc[D2,M,N]) * kernel[M,N]
  reshape(out, (d*m,d*n))
end