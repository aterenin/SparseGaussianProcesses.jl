import Flux
using LinearAlgebra
using Statistics
import Random.rand!
import Random.randn!

export SparseGaussianProcess, rand!

abstract type GaussianProcess end

function rand!(gp::GaussianProcess; num_samples::Integer = num_samples(gp))
  # sample prior weights
  s = num_samples
  (l,_) = size(gp.prior_weights)
  w = randn!(similar(gp.prior_weights, l, s))

  # build kernel matrix
  (z,mu,U,V) = gp.inducing_points()
  (k,B) = (gp.kernel, gp.inter_domain_operator)
  K = (B*k*B)(z,z)
  Q = cholesky(K + V'*V)

  # sample data weights given prior weights
  dm = length(mu)
  u = U isa Nothing ? zero(eltype(mu)) : U' * randn!(similar(mu, (dm, s)))
  e = V' * randn!(similar(mu, (dm, s)))
  f = reshape(gp.prior_basis(z, w, k, B), (dm, s))
  v = mu .+ Q \ (u .- f .- e)

  gp.prior_weights = w
  gp.inducing_weights = v
  gp.cholesky_cache = Q
  nothing
end

function (gp::GaussianProcess)(x::AbstractMatrix)
  (id,n) = size(x)
  (w,v) = (gp.prior_weights, gp.inducing_weights)
  (z,_,_,_) = gp.inducing_points()
  (k,A,B) = (gp.kernel, gp.observation_operator, gp.inter_domain_operator)
  s = num_samples(gp)

  # evaluate prior part at x
  f_prior = gp.prior_basis(x, w, k, A)

  # evaluate data part at x
  (_,od) = (A*k*B).dims
  K = (A*k*B)(x, z)
  f_data = reshape(K * v, (od,n,s)) # non-batched

  # combine
  f_prior .+ f_data
end

function num_samples(gp::GaussianProcess)
  size(gp.inducing_weights, ndims(gp.inducing_weights))
end



mutable struct SparseGaussianProcess{
    K<:CovarianceKernel,
    OO<:InterDomainOperator,
    IO<:InterDomainOperator,
    RF<:RandomFeatures,
    IP<:InducingPoints,
    M<:AbstractMatrix,
    V<:AbstractVector,
    C<:Cholesky,
    H<:Hyperprior
    } <: GaussianProcess
  kernel                :: K
  observation_operator  :: OO
  inter_domain_operator :: IO
  prior_basis           :: RF
  prior_weights         :: M
  inducing_points       :: IP
  inducing_weights      :: M
  log_error             :: V
  cholesky_cache        :: C
  hyperprior            :: NamedTuple{(:log_error,), Tuple{H}}
end

Flux.trainable(gp::SparseGaussianProcess) = ()
Flux.@functor SparseGaussianProcess

function SparseGaussianProcess(k::CovarianceKernel)
  (id,od) = k.dims
  oo = IdentityOperator()
  io = IdentityOperator()
  pb = EuclideanRandomFeatures(k, 64)
  pw = zeros(64,1)
  db = MarginalInducingPoints(k, od, 10)
  dw = zeros(10,1)
  le = zeros(1)
  cc = k(db.location, db.location)|>cholesky
  hp = (log_error = NormalHyperprior([0.],[1.]),)
  SparseGaussianProcess(k,oo,io,pb,pw,db,dw,le,cc,hp)
end

# function SPGaussianProcess(k::K; m::Integer = 10, l::Integer = 64) where K<:Kernel{Fl,D} where {Fl<:AbstractFloat, D}
#   z = rand(D, m)
#   mu = zeros(m)
#   U = Diagonal(ones(m)) ./ 10
#   omega = zeros(D, l)
#   beta = zeros(l)
#   log_sigma = [log(0.1)]
#   jitter = 0.000001
#   v = zeros(m,1)
#   w = zeros(l,1)
#   hyperprior = (log_sigma = NormalHyperprior([2*log(0.1)],[1.0]),)
#   gp = SPGaussianProcess{Fl,Vector{Fl},Matrix{Fl}, K}(k, z, mu, U, omega, beta, log_sigma, jitter, v, w, hyperprior)
#   resample_basis!(gp)
#   gp
# end

