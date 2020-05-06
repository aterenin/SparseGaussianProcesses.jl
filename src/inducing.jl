using LinearAlgebra

export MarginalInducingPoints

"""
    InducingPoints

An abstract set of inducing points.
"""
abstract type InducingPoints end
  
"""
    MarginalInducingPoints

A set of inducing points representing the marginal value of the 
Gaussian process ``u = (\\mathcal{B} g)(z)``.
"""
mutable struct MarginalInducingPoints{V<:AbstractVector, M<:AbstractMatrix, C<:Cholesky} <: InducingPoints
  location            :: M
  mean                :: V
  covariance_triangle :: M
  log_jitter          :: V
  weights             :: M
  cholesky_cache      :: C
end

Flux.trainable(ip::MarginalInducingPoints) = (ip.location, ip.mean, ip.covariance_triangle)
Flux.@functor MarginalInducingPoints
Flux.@functor Cholesky

"""
    MarginalInducingPoints(k::CovarianceKernel, num_inducing::Integer)

Creates a set of marginal inducing points with covariance kernel ``k`` of ``u``.
"""
function MarginalInducingPoints(k::CovarianceKernel, num_inducing::Int)
  (id,od) = k.dims
  location = randn(id, num_inducing)
  mean = zeros(od*num_inducing)
  covariance_triangle = zeros(od*num_inducing, od*num_inducing)
  log_jitter = log.([0.00001])./2
  weights = zeros(num_inducing,1)
  cholesky_cache = cholesky(k(location,location) + Diagonal(I(od*num_inducing) * exp.(log_jitter .* 2)))
  ip = MarginalInducingPoints(location, mean, covariance_triangle, log_jitter, weights, cholesky_cache)
  ip(location, k)
  ip
end

"""
    (self::MarginalInducingPoints)(z::AbstractMatrix, k::Kernel)

Sets the inducing locations of ``u`` to ``z``, inducing mean to zero, and
inducing covariance to ``k(z,z)``.
"""
function (self::MarginalInducingPoints)(z::AbstractMatrix, k::Kernel)
  (id,od) = k.dims
  n_inducing = size(z,ndims(z))
  self.location = z
  self.mean = (similar(self.mean, (od*n_inducing)) .= 0)
  K = k(self.location,self.location)
  D = Diagonal(I(size(K,1)) * exp.(self.log_jitter .* 2))
  Q = cholesky(K + D)
  self.covariance_triangle = copy(Q.U)
  self.covariance_triangle[diagind(self.covariance_triangle)] .= log.(diag(self.covariance_triangle))
  self.weights = (similar(self.weights, (n_inducing, size(self.weights,ndims(self.weights)))) .= 0)
  self.cholesky_cache = Q
  nothing
end

"""
    (self::MarginalInducingPoints)()

Assembles the inducing covariance into upper-triangular form, with diagonal 
values exponentiated to ensure they are positive. Returns inducing locations, 
inducing mean, jitter covariance, and inducing covariance.
"""
function (self::MarginalInducingPoints)()
  ones = (randn!(similar(self.log_jitter, (size(self.covariance_triangle,1)))) .* 0 .+ 1) # HACK: suppress autodiff unsupported mutation error without NaNs
  U = UnitUpperTriangular(self.covariance_triangle) + Diagonal(exp.(diag(self.covariance_triangle)) .- ones)
  D = Diagonal(ones .* exp.(self.log_jitter))
  (self.location, self.mean, U, D, self.cholesky_cache)
end


"""
    PseudoDataInducingPoints

A set of inducing points representing pseudo-data points with diagonal 
non-constant error covariance.
"""
mutable struct PseudoDataInducingPoints{V<:AbstractVector,M<:AbstractMatrix,C<:Cholesky} <: InducingPoints
  location                :: M
  mean                    :: V
  log_covariance_diagonal :: V
  weights                 :: M
  cholesky_cache          :: C
end

"""
    PseudoDataInducingPoints(k::Kernel, dim::Int, num_inducing::Int)

Creates a set of pseudo-data inducing points with unit error variance.
"""
function PseudoDataInducingPoints(k::Kernel, dim::Int, num_inducing::Int)
  (id,od) = k.dims
  location = randn(id, num_inducing)
  mean = zeros(dim*num_inducing)
  log_covariance_diagonal = zeros(dim*num_inducing)
  weights = zeros(num_inducing,1)
  cholesky_cache = cholesky(k(location,location) .+ Diagonal(exp.(log_covariance_diagonal)))
  PseudoDataInducingPoints(location, mean, log_covariance_diagonal, weights, cholesky_cache)
end

"""
    (self::PseudoDataInducingPoints)(z::AbstractMatrix, k::Kernel)

Sets the inducing locations of ``u`` to ``z``, inducing mean to zero, and 
inducing error variance to one for each pseudo-data point.
"""
function (self::PseudoDataInducingPoints)(z::AbstractMatrix, k::Kernel)
  (id,od) = k.dims
  n_inducing = size(z,ndims(z))
  self.location = z
  self.mean = (similar(self.mean, (od*n_inducing)) .= 0)
  self.log_covariance_diagonal = log.(diag(k(self.location,self.location)))
  self.weights = zeros(size(z,ndims(z)),size(self.weights,ndims(self.weights)))
  self.cholesky_cache = cholesky(k(self.location,self.location) + Diagonal(exp.(self.log_covariance_diagonal)))
  nothing
end

"""
    (self::PseudoDataInducingPoints)()

Assembles the pseudo-data inducing error variance into matrix form. Returns 
inducing locations, inducing mean, `nothing` jitter term, and inducing error 
variance.
"""
function (self::PseudoDataInducingPoints)()
  D = Diagonal(exp.(self.log_covariance_diagonal))
  (self.location, self.mean, nothing, D)
end