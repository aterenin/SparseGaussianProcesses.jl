import LinearAlgebra
import Flux
import Flux.CuArrays
import Flux.CuArrays.GPUArrays
import Base.+
import Random.randn!

Flux.@nograd randn! # CuArrays workaround
Flux.@nograd Base.similar # CuArrays workaround

function Base.copyto!(x::C, y::UpperTriangular{Fl,C}) where {Fl<:AbstractFloat, C<:GPUArrays.AbstractGPUArray{Fl}}
  copyto!(x, y.data)
end

function Base.copyto!(x::UpperTriangular{Fl,C}, y::UpperTriangular{Fl,C}) where {Fl<:AbstractFloat, C<:GPUArrays.AbstractGPUArray{Fl}}
  copyto!(x.data, y.data)
end

function +(x::LinearAlgebra.UpperTriangular{Fl,C}, y::LinearAlgebra.Diagonal{Fl,CV}) where {Fl<:AbstractFloat, C<:CuArrays.CuArray{Fl,2}, CV<:CuArrays.CuArray{Fl,1}} 
  LinearAlgebra.UpperTriangular(x .+ y)
end

function Base.inv(x::LinearAlgebra.UpperTriangular{Fl,C}) where {Fl<:AbstractFloat, C<:CuArrays.CuArray{Fl,2}} 
  LinearAlgebra.UpperTriangular(x \ CuArrays.CuArray(one(Fl)*LinearAlgebra.I,size(x)...))
end

function LinearAlgebra.BLAS.trmm!(side::Char, uplo::Char, transa::Char, diag::Char, alpha::Fl, A::C, B::C) where {Fl<:AbstractFloat, C<:CuArrays.CuArray{Fl,2}} 
  CuArrays.CUBLAS.trmm!(side, uplo, transa, diag, alpha, A, B, B)
end

function LinearAlgebra.BLAS.trsm!(side::Char, uplo::Char, transa::Char, diag::Char, alpha::Float32, A::C, B::C) where {Fl<:AbstractFloat, C<:CuArrays.CuArray{Fl,2}} 
  CuArrays.CUBLAS.trsm!(side, uplo, transa, diag, alpha, A, B)
end

function LinearAlgebra.diag(x::LinearAlgebra.Adjoint{Fl,C}, i::Int) where {Fl<:AbstractFloat, C<:CuArrays.CuArray{Fl,2}} 
  diag(x.parent, -i)
end

function +(x::LinearAlgebra.UnitUpperTriangular{Fl,LinearAlgebra.Adjoint{Fl,C}}, y::LinearAlgebra.UniformScaling) where {Fl<:AbstractFloat, C<:CuArrays.CuArray{Fl,2}}
  LinearAlgebra.LowerTriangular(x.data.parent + y)'
end

function LinearAlgebra.mul!(X::C, Y::C, Z::LinearAlgebra.Adjoint{Fl,LinearAlgebra.UpperTriangular{Fl,C}}) where {Fl<:AbstractFloat, C<:CuArrays.CuArray{Fl,2}}
  X .= Y * Z
end


module UniformScalingGPU

using LinearAlgebra
using Flux.CuArrays.GPUArrays
using Flux.CuArrays.GPUArrays: AbstractGPUMatrix

import Base: +, -

const genericwrappers = (
    :LowerTriangular,
    :UpperTriangular,
    :Hermitian,
    :Symmetric
)

const unittriangularwrappers = (
    (:UnitUpperTriangular, :UpperTriangular), 
    (:UnitLowerTriangular, :LowerTriangular)
)

function kernel_generic(ctx, B, J)
    @inbounds index = diagind(B)[linear_index(ctx)]
    @inbounds B[index] += J
    return nothing
end

function kernel_unittriangular(ctx, B, J, diagonal_val)
    @inbounds index = diagind(B)[linear_index(ctx)]
    @inbounds B[index] = diagonal_val + J
    return nothing
end

for (t1, t2) in unittriangularwrappers
    @eval begin
        function (+)(A::$t1{T, <:AbstractGPUMatrix}, J::UniformScaling) where T
            B = similar(parent(A), typeof(oneunit(T) + J))
            copyto!(B, parent(A))
            gpu_call(kernel_unittriangular, B, J, one(eltype(B)); total_threads=minimum(size(B)))
            return $t2(B)
        end

        function (-)(J::UniformScaling, A::$t1{T, <:AbstractGPUMatrix}) where T
            B = similar(parent(A), typeof(J - oneunit(T)))
            B .= .- parent(A)
            gpu_call(kernel_unittriangular, B, J, -one(eltype(B)); total_threads=minimum(size(B)))
            return $t2(B)
        end
    end
end

for t in genericwrappers
    @eval begin
        function (+)(A::$t{T, <:AbstractGPUMatrix}, J::UniformScaling) where T
            B = similar(parent(A), typeof(oneunit(T) + J))
            copyto!(B, parent(A))
            gpu_call(kernel_generic, B, J; total_threads=minimum(size(B)))
            return $t(B)
        end

        function (-)(J::UniformScaling, A::$t{T, <:AbstractGPUMatrix}) where T
            B = similar(parent(A), typeof(J - oneunit(T)))
            B .= .- parent(A)
            gpu_call(kernel_generic, B, J; total_threads=minimum(size(B)))
            return $t(B)
        end
    end
end

# Breaking Hermiticity when adding a complex value to the diagonal
function (+)(A::Hermitian{T,<:AbstractGPUMatrix}, J::UniformScaling{<:Complex}) where T
    B = similar(parent(A), typeof(oneunit(T) + J))
    copyto!(B, parent(A))
    gpu_call(kernel_generic, B, J; total_threads=minimum(size(B)))
    return B
end

function (-)(J::UniformScaling{<:Complex}, A::Hermitian{T,<:AbstractGPUMatrix}) where T
    B = similar(parent(A), typeof(J - oneunit(T)))
    B .= .-parent(A)
    gpu_call(kernel_generic, B, J; total_threads=minimum(size(B)))
    return B
end

# Finally the generic matrix version
function (+)(A::AbstractGPUMatrix{T}, J::UniformScaling) where T
    B = similar(A, typeof(oneunit(T) + J))
    copyto!(B, A)
    gpu_call(kernel_generic, B, J; total_threads=minimum(size(B)))
    return B
end

function (-)(J::UniformScaling, A::AbstractGPUMatrix{T}) where T
    B = similar(A, typeof(J - oneunit(T)))
    B .= .-A
    gpu_call(kernel_generic, B, J; total_threads=minimum(size(B)))
    return B
end

end # module