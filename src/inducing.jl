using LinearAlgebra

export MarginalInducingPoints

abstract type InducingPoints end
  
mutable struct MarginalInducingPoints{V<:AbstractVector, M<:AbstractMatrix, C<:Cholesky} <: InducingPoints
  location            :: M
  mean                :: V
  covariance_triangle :: M
  log_jitter          :: V
  weights             :: M
  cholesky_cache      :: C
end

function MarginalInducingPoints(k::CovarianceKernel, num_inducing::Integer)
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

function (self::MarginalInducingPoints)()
  U = UnitUpperTriangular(self.covariance_triangle) - I + Diagonal(exp.(diag(self.covariance_triangle)))
  D = Diagonal(I(size(self.covariance_triangle,1)) * exp.(self.log_jitter))
  (self.location, self.mean, U, D, self.cholesky_cache)
end


mutable struct PseudoDataInducingPoints{V<:AbstractVector,M<:AbstractMatrix,C<:Cholesky} <: InducingPoints
  location                :: M
  mean                    :: V
  log_covariance_diagonal :: V
  weights                 :: M
  cholesky_cache          :: C
end

function PseudoDataInducingPoints(k::Kernel, dim::Integer, num_inducing::Integer)
  (id,od) = k.dims
  location = randn(id, num_inducing)
  mean = zeros(dim*num_inducing)
  log_covariance_diagonal = zeros(dim*num_inducing)
  weights = zeros(num_inducing,1)
  cholesky_cache = cholesky(k(location,location) .+ Diagonal(exp.(log_covariance_diagonal)))
  PseudoDataInducingPoints(location, mean, log_covariance_diagonal, weights, cholesky_cache)
end

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

function (self::PseudoDataInducingPoints)()
  D = Diagonal(exp.(self.log_covariance_diagonal))
  (self.location, self.mean, nothing, D)
end