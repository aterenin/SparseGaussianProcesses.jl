import Base.*

"""
    Kernel

An abstract kernel.
"""
abstract type Kernel end 

"""
    CovarianceKernel

A covariance kernel, assumed symmetric in its arguments.
"""
abstract type CovarianceKernel <: Kernel end

"""
    CrossCovarianceKernel

A cross-covariance kernel, NOT necessarily symmetric in its arguments.
"""
abstract type CrossCovarianceKernel <: Kernel end

"""
    pairwise_column_difference(x::AbstractMatrix, y::AbstractMatrix)

Computes the 3-dimensional distance array, where out dimensions 
are `(input_dimension, x_data_point_dimension, y_data_point_dimension)`.
"""
function pairwise_column_difference(x::AbstractMatrix, y::AbstractMatrix)
  (d1,m) = size(x)
  (d2,n) = size(y)
  reshape(x,(d1,m,1)) .- reshape(y,(d2,1,n))
end

"""
    EuclideanKernel

An abstract covariance kernel defined over a Euclidean space.
"""
abstract type EuclideanKernel <: CovarianceKernel end

include("kernels/operators.jl")
include("kernels/euclidean.jl")
include("kernels/circular.jl")


"""
    (k::Kernel)(x1::AbstractArray{<:Any,3}, x2::AbstractMatrix)

Computes a kernel matrix in batched form.
"""
function (k::Kernel)(x1::AbstractArray{<:Any,3}, x2::AbstractMatrix)
  (d,n1,n2) = size(x1)
  x1r = reshape(x1, (d,n1*n2))
  k(x1r, x2)
end