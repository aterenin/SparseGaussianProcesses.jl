Flux.@nograd Base.similar
Base.copyto!(A::UpperTriangular{<:Any, <:Flux.CuArrays.GPUArrays.AbstractGPUArray}, B::UpperTriangular{<:Any, <:Flux.CuArrays.GPUArrays.AbstractGPUArray}) = LinearAlgebra.triu!(copyto!(parent(A), parent(B)))

module GPU

using LinearAlgebra
using Flux.CuArrays: CuMatrix
using Flux.CuArrays.CUBLAS: CublasFloat
import Base: +

function LinearAlgebra.diag(x::Adjoint{T,<:CuMatrix{T}}, i::Int) where T<:CublasFloat
  diag(parent(x), -i)
end

function +(x::UnitUpperTriangular{T,<:Adjoint{T,<:CuMatrix{T}}}, y::UniformScaling) where T<:CublasFloat
  LowerTriangular(parent(parent(x)) + y)'
end

end # module