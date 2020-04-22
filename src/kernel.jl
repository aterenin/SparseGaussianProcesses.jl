import Base.*

abstract type Kernel end 
abstract type CovarianceKernel <: Kernel end
abstract type CrossCovarianceKernel <: Kernel end

function pairwise_column_difference(x::AbstractMatrix, y::AbstractMatrix)
  (d1,m) = size(x)
  (d2,n) = size(y)
  reshape(x,(d1,m,1)) .- reshape(y,(d2,1,n))
end

include("kernels/euclidean.jl")



abstract type InterDomainOperator end

export IdentityOperator, GradientOperator

struct IdentityOperator <: InterDomainOperator end
struct GradientOperator <: InterDomainOperator end

*(k::Kernel, op::IdentityOperator) = k
*(op::IdentityOperator, k::Kernel) = k