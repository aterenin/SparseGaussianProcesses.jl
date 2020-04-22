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