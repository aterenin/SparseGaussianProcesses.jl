export MarginalInducingPoints

abstract type InducingPoints end
  
mutable struct MarginalInducingPoints{V<:AbstractVector, M<:AbstractMatrix} <: InducingPoints
  location            :: M
  mean                :: V
  covariance_triangle :: M
  log_jitter          :: V
end

function MarginalInducingPoints(k::CovarianceKernel, dim::Integer, num_inducing::Integer)
  (id,od) = k.dims
  location = randn(id, num_inducing)
  mean = zeros(dim*num_inducing)
  covariance_triangle = zeros(dim*num_inducing, dim*num_inducing)
  log_jitter = log.([0.00001])./2
  MarginalInducingPoints(location, mean, covariance_triangle, log_jitter)
end

function (self::MarginalInducingPoints)(z::AbstractMatrix, k::Kernel)
  self.location = z
  self.mean .*= 0
  K = k(self.location,self.location)
  self.covariance_triangle .= cholesky(K).U
  self.covariance_triangle[diagind(self.covariance_triangle)] .= log.(diag(self.covariance_triangle))
  nothing
end

function (self::MarginalInducingPoints)()
  U = UnitUpperTriangular(self.covariance_triangle) - I + Diagonal(exp.(diag(self.covariance_triangle)))
  D = Diagonal(I(size(self.covariance_triangle,1)) .* exp.(self.log_jitter))
  (self.location, self.mean, U, D)
end


mutable struct PseudoDataInducingPoints{V<:AbstractVector,M<:AbstractMatrix} <: InducingPoints
  location                :: M
  mean                    :: V
  log_covariance_diagonal :: V
end

function PseudoDataInducingPoints(k::Kernel, dim::Integer, num_inducing::Integer)
  (id,od) = k.dims
  location = randn(id, num_inducing)
  mean = zeros(dim*num_inducing)
  log_covariance_diagonal = zeros(dim*num_inducing)
  PseudoDataInducingPoints(location, mean, log_covariance_diagonal)
end

function (self::PseudoDataInducingPoints)(z::AbstractMatrix, k::Kernel)
  self.location = z
  self.mean .*= 0
  self.log_covariance_diagonal .= log.(k.(self.location))
  nothing
end

function (self::PseudoDataInducingPoints)()
  D = Diagonal(exp.(self.log_covariance_diagonal))
  (self.location, self.mean, nothing, D)
end