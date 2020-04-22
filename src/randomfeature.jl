import Random.rand!
using TensorCast

export EuclideanRandomFeatures

abstract type RandomFeatures end



mutable struct EuclideanRandomFeatures{A<:AbstractArray{<:Any,3},M<:AbstractMatrix} <: RandomFeatures
  frequency  :: A
  phase      :: M
  weights    :: M
end

function rand!(self::EuclideanRandomFeatures, k::EuclideanKernel, num_features::Integer = size(self.frequency,ndims(self.frequency)))
  (id,od) = k.dims
  Fl = eltype(self.frequency)
  self.frequency = spectral_distribution(k, num_features)
  self.phase = Fl(2*pi) .* rand!(similar(self.phase, (od,num_features)))
  nothing
end

function EuclideanRandomFeatures(k::EuclideanKernel, num_features::Integer)
  (id,od) = k.dims
  frequency = zeros(id,od,num_features)
  phase = zeros(od,num_features)
  weights = zeros(num_features,1)
  features = EuclideanRandomFeatures(frequency, phase, weights)
  rand!(features, k)
  features
end

function (self::EuclideanRandomFeatures)(x::AbstractMatrix, w::AbstractMatrix, k::EuclideanKernel, op::IdentityOperator)
  Fl = eltype(self.frequency)
  l = size(self.frequency, ndims(self.frequency))

  # (_,n) = size(x)
  # (id,od,l) = size(self.frequency)
  # (_,s) = size(w)
  # rescaled_freq = reshape(exp.(k.log_length_scales ./ -2), (id,1,1)) .* self.frequency
  # basis_fn_inner_prod = dropdims(sum(reshape(rescaled_freq, (id,od,l,1)) .* reshape(x, (id,1,1,n)); dims=1); dims=1)
  # basis_fn = cos.(basis_fn_inner_prod .+ reshape(self.phase, (od,l,1)))
  # basis_weight = sqrt(Fl(2)) .* exp.(k.log_variance ./ 2) ./ sqrt(Fl(l)) .* w
  # dropdims(sum(reshape(basis_fn,(od,l,n,1)) .* reshape(basis_weight, (1,l,1,s)); dims=2); dims=2) # non-batched

  # @cast rescaled_freq[ID,OD,L] := self.frequency[ID,OD,L] / exp(k.log_length_scales[ID])
  # @matmul basis_fn_inner_prod[OD,L,N] := sum(ID) rescaled_freq[ID,OD,L] * x[ID,N]
  @cast rescaled_x[ID,N] := x[ID,N] / exp(k.log_length_scales[ID])
  @matmul basis_fn_inner_prod[OD,L,N] := sum(ID) self.frequency[ID,OD,L] * rescaled_x[ID,N]
  @cast basis_fn[OD,L,N] := cos(basis_fn_inner_prod[OD,L,N] + self.phase[OD,L])
  basis_weight = sqrt(Fl(2)) .* exp.(k.log_variance ./ 2) ./ sqrt(Fl(l)) .* w
  @matmul output[OD,N,S] := sum(L) basis_fn[OD,L,N] * basis_weight[L,S]
  output
end

# function (self::ScalarRandomFeatures)(x::AbstractMatrix, w::AbstractMatrix, k::EuclideanKernel{D}, op::GradientOperator; dims::UnitRange=1:D) where D
#   Fl = eltype(self.frequency)
#   (_,l) = size(self.frequency)
#   (_,n) = size(x)
#   d = length(dims)
#   frequency_ls = exp.(self.kernel.log_length_scales ./ -2) .* self.frequency
#   basis_fn_grad_outer = -sin.(self.frequency_ls' * x .+ self.phase)
#   basis_fn_grad = reshape(basis_fn_grad_outer, (1,l,n)) .* reshape(view(self.frequency_ls, :,:), (d,l,1))
#   basis_weight = sqrt(Fl(2)) .* exp.(self.kernel.log_variance ./ 2) ./ sqrt(Fl(l)) .* w
#   dropdims(sum(reshape(basis_weight,(1,l,1,:)) .* reshape(basis_fn_grad, (d,l,n,1)); dims=2); dims=2) # non-batched
# end

# function (self::ScalarRandomFeatures)(x::AbstractArray{<:Any,3}, w::AbstractMatrix, k::EuclideanKernel{D}, op::GradientOperator; dims::UnitRange=1:D) where D
#   Fl = eltype(self.frequency)
#   (_,l) = size(self.frequency)
#   (_,n1,n2) = size(x)
#   d = length(dims)
#   frequency_ls = exp.(self.kernel.log_length_scales ./ -2) .* self.frequency
#   basis_fn_grad_outer = -sin.(self.frequency_ls' * reshape(x, (:, n1*n2)) .+ self.phase)
#   basis_fn_grad = reshape(basis_fn_grad_outer, (1,l,n1,n2)) .* reshape(view(self.frequency_ls, :,:), (d,l,1,1))
#   basis_weight = sqrt(Fl(2)) .* exp.(self.kernel.log_variance ./ 2) ./ sqrt(Fl(l)) .* w
#   dropdims(sum(reshape(basis_weight,(1,l,1,:)) .* basis_fn_grad; dims=2); dims=2) # batched
# end