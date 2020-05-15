import Random.rand!
using TensorCast

export EuclideanRandomFeatures

"""
    EuclideanRandomFeatures

A set of Euclidean random features, parameterized by frequency and phase,
together with a set of associated basis weights.
"""
mutable struct EuclideanRandomFeatures{A<:AbstractArray{<:Any,3},M<:AbstractMatrix} <: PriorBasis
  frequency  :: A
  phase      :: M
  weights    :: M
end

Flux.trainable(f::EuclideanRandomFeatures) = ()
Flux.@functor EuclideanRandomFeatures

"""
    rand!(self::EuclideanRandomFeatures, k::EuclideanKernel, 
          num_features::Int = size(self.frequency,ndims(self.frequency)))

Draw a new set of random features, by randomly sampling a new frequencies
from the spectral measure, and new phases uniformly from ``(0, 2\\pi)``.
Does NOT automatically resample the GP containing the features.
"""
function rand!(self::EuclideanRandomFeatures, k::EuclideanKernel, num_features::Int = size(self.frequency,ndims(self.frequency)))
  (id,od) = k.dims
  (_,s) = size(self.weights)
  Fl = eltype(self.frequency)
  self.frequency = spectral_distribution(k, num_features)
  self.phase = Fl(2*pi) .* rand!(similar(self.phase, (od,num_features)))
  self.weights = randn!(similar(self.weights, (num_features, s)))
  nothing
end

"""
    EuclideanRandomFeatures(k::EuclideanKernel, num_features::Int)

Create a set of Euclidean random features with eigenvalues given by the 
spectral distribution given by ``k``.
"""
function EuclideanRandomFeatures(k::EuclideanKernel, num_features::Int)
  (id,od) = k.dims
  frequency = zeros(id,od,num_features)
  phase = zeros(od,num_features)
  weights = zeros(num_features,1)
  features = EuclideanRandomFeatures(frequency, phase, weights)
  rand!(features, k)
  features
end

"""
    (self::EuclideanRandomFeatures)(x::AbstractMatrix, w::AbstractMatrix, 
                                    k::EuclideanKernel)

Evaluate the ``f(x)`` where ``f`` is a Gaussian process with kernel ``k``, 
and ``x`` is the data, using the random features.
"""
function (self::EuclideanRandomFeatures)(x::AbstractMatrix, k::EuclideanKernel)
  Fl = eltype(self.frequency)
  l = size(self.frequency, ndims(self.frequency))
  @cast rescaled_x[ID,N] := x[ID,N] / exp(k.log_length_scales[ID])
  @matmul basis_fn_inner_prod[OD,L,N] := sum(ID) self.frequency[ID,OD,L] * rescaled_x[ID,N]
  @cast basis_fn[OD,L,N] := cos(basis_fn_inner_prod[OD,L,N] + self.phase[OD,L])
  basis_weight = sqrt(Fl(2)) .* exp.(k.log_variance ./ 2) ./ sqrt(Fl(l)) .* self.weights
  @matmul output[OD,N,S] := sum(L) basis_fn[OD,L,N] * basis_weight[L,S]
  output
end

"""
    (self::EuclideanRandomFeatures)(x::AbstractMatrix, w::AbstractMatrix, 
                                    k::GradientKernel{<:EuclideanKernel})

Evaluate ``(\\nabla g)(x)`` where ``g`` is a Gaussian process with kernel ``k``,
``\\nabla`` is the gradient inter-domain operator, and ``x`` is the data, using
the random features.
"""
function (self::EuclideanRandomFeatures)(x::AbstractMatrix, k::GradientKernel{<:EuclideanKernel})
  Fl = eltype(self.frequency)
  l = size(self.frequency, ndims(self.frequency))
  @cast rescaled_x[ID,N] := x[ID,N] / exp(k.parent.log_length_scales[ID])
  @matmul basis_fn_inner_prod[OD,L,N] := sum(ID) self.frequency[ID,OD,L] * rescaled_x[ID,N]
  @cast basis_fn_grad_outer[OD,L,N] := -sin(basis_fn_inner_prod[OD,L,N] + self.phase[OD,L])
  @cast basis_fn_grad[ID,OD,L,N] := basis_fn_grad_outer[OD,L,N] * self.frequency[ID,OD,L] / exp(k.parent.log_length_scales[ID])
  basis_weight = sqrt(Fl(2)) .* exp.(k.parent.log_variance ./ 2) ./ sqrt(Fl(l)) .* self.weights
  @matmul output[ID,OD,N,S] := sum(L) basis_fn_grad[ID,OD,L,N] * basis_weight[L,S]
  dropdims(output; dims=2)
end