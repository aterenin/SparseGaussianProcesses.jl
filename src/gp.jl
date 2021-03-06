import Flux
using LinearAlgebra
using Statistics
import Random.rand!
import Random.randn!

export SparseGaussianProcess, rand!

"""
    GaussianProcess
    
An abstract Gaussian process type.
"""
abstract type GaussianProcess end


"""
    rand!(gp::GaussianProcess; num_samples = num_samples(gp))

Draw a new set of samples from the Gaussian process.
"""
function rand!(gp::GaussianProcess; num_samples::Int = num_samples(gp))
  # sample prior weights
  s = num_samples
  (l,_) = size(gp.prior_basis.weights)
  w = randn!(similar(gp.prior_basis.weights, (l, s)))

  gp.prior_basis.weights = w

  # build kernel matrix
  (z,mu,U,V) = gp.inducing_points()
  (k,B) = (gp.kernel, gp.inter_domain_operator)
  K = (B*k*B)(z,z)
  Q = cholesky(K + V'*V)

  # sample data weights given prior weights
  dm = length(mu)
  u = U isa Nothing ? zero(eltype(mu)) : U' * randn!(similar(mu, (dm, s)))
  e = V' * randn!(similar(mu, (dm, s)))
  f = reshape(gp.prior_basis(z, B*k*B), (dm, s))
  v = mu .+ Q \ (u .- f .- e)

  gp.inducing_points.weights = v
  gp.inducing_points.cholesky_cache = Q
  nothing
end

"""
    (gp::GaussianProcess)(x::AbstractMatrix)

Evaluate `gp` at a set of points `x`. Returns a 3-array whose dimensions
are ordered `(output_dimension, data_point_index, sample_index)`.
"""
function (gp::GaussianProcess)(x::AbstractMatrix)
  (id,n) = size(x)
  (z,_,_,_) = gp.inducing_points()
  (k,A,B) = (gp.kernel, gp.observation_operator, gp.inter_domain_operator)
  s = num_samples(gp)

  # evaluate prior part at x
  f_prior = gp.prior_basis(x, A*k*A)

  # evaluate data part at x
  (_,od) = (A*k*B).dims
  K = (A*k*B)(x, z)
  v = gp.inducing_points.weights
  f_data = reshape(K * v, (od,n,s)) # non-batched

  # combine
  f_prior .+ f_data
end

"""
    (gp::GaussianProcess)(x::AbstractArray{<:Any,3})

Evaluate `gp` at a batched set of points `x`. Returns a 3-array whose dimensions
are ordered `(output_dimension, data_point_index, batch_index)`.
"""
function (gp::GaussianProcess)(x::AbstractArray{<:Any,3})
  (id,n,b) = size(x)
  (z,_,_,_) = gp.inducing_points()
  (k,A,B) = (gp.kernel, gp.observation_operator, gp.inter_domain_operator)
  s = num_samples(gp)

  # evaluate prior part at x
  f_prior = gp.prior_basis(x, A*k*A)

  # evaluate data part at x
  (_,od) = (A*k*B).dims
  K = reshape((A*k*B)(x, z), (od,n,b,:))
  v = reshape(gp.inducing_points.weights', (1,1,s,:))
  f_data = dropdims(sum(K .* v; dims=4); dims=4) # batched

  # combine
  f_prior .+ f_data
end

"""
    num_samples(gp::GaussianProcess)

Returns the number of random samples currently stored in `gp`.
"""
function num_samples(gp::GaussianProcess)
  size(gp.inducing_points.weights, ndims(gp.inducing_points.weights))
end


"""
    SparseGaussianProcess

A sparse Gaussian process, with kernel `kernel`.
It supports models of the form

``(f | u)(\\cdot) = (\\mathcal{A} g)(\\cdot) + K_{(\\cdot)z} 
                (K_{zz} + \\Lambda)^{-1} (u - (\\mathcal{B} g)(z) - \\epsilon)``

where ``g \\sim GP(0, k)``, ``u \\sim N(\\mu, \\Sigma)``, 
``\\epsilon \\sim N(0, \\Lambda)``, and ``\\mathcal{A}``, ``\\mathcal{B}`` 
are inter-domain operators.

# Fields
- `kernel`: the kernel ``k`` of ``g``.
- `observation_operator`: the observation operator ``\\mathcal{A}``.
- `inter_domain_operator`: the inter-domain operator ``\\mathcal{B}``.
- `prior_basis`: the basis and weights used for efficiently sampling the prior.
- `inducing_points`: the basis and weights used for sampling the data-dependent 
  portion of the GP.
- `log_error`: the error variance (log-scale, trainable by default).
- `hyperprior`: the hyperprior used for the log error term.
"""
mutable struct SparseGaussianProcess{
    K<:CovarianceKernel,
    OO<:InterDomainOperator,
    IO<:InterDomainOperator,
    PB<:PriorBasis,
    IP<:InducingPoints,
    V<:AbstractVector,
    H<:Hyperprior
    } <: GaussianProcess
  kernel                :: K
  observation_operator  :: OO
  inter_domain_operator :: IO
  prior_basis           :: PB
  inducing_points       :: IP
  log_error             :: V
  hyperprior            :: NamedTuple{(:log_error,), Tuple{H}}
end

Flux.@functor SparseGaussianProcess

"""
    SparseGaussianProcess(k::CovarianceKernel)

Creates a `SparseGaussianProcess` with kernel `k`, with 64 random features 
and 10 inducing points by default.
"""
function SparseGaussianProcess(k::CovarianceKernel; 
  observation_operator = IdentityOperator(), 
  inter_domain_operator = IdentityOperator(),
  prior_basis = EuclideanRandomFeatures,
  inducing_points = MarginalInducingPoints,
  log_error = zeros(1),
  hyperprior = (log_error = NormalHyperprior([0.],[1.]),)
  )
  SparseGaussianProcess(k,observation_operator,inter_domain_operator,prior_basis(k,64),inducing_points(inter_domain_operator*k*inter_domain_operator, 11),log_error,hyperprior)
end
