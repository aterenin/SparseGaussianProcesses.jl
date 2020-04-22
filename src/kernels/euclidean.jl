import Flux
using Random: randn!

export SquaredExponentialKernel

abstract type EuclideanKernel <: CovarianceKernel end

function hyperprior_logpdf(self::EuclideanKernel)
  self.hyperprior.log_variance(self.log_variance) .+ 
    self.hyperprior.log_length_scales(self.log_length_scales)
end

Flux.trainable(k::EuclideanKernel) = (k.log_length_scales,k.log_variance)



struct SquaredExponentialKernel{V<:AbstractVector,H<:Hyperprior} <: EuclideanKernel
  dims              :: Tuple{Integer, Integer}
  log_variance      :: V
  log_length_scales :: V
  hyperprior        :: NamedTuple{(:log_variance, :log_length_scales), Tuple{H,H}}
end

# Flux.functor(k::SquaredExponentialKernel{Fl,D,V}) where {Fl,D,V} = ((log_variance = k.log_variance, log_length_scales = k.log_length_scales, hyperprior = k.hyperprior), x->SquaredExponentialKernel{eltype(x[1]),D,typeof(x[1])}(x[1],x[2],x[3]))

function SquaredExponentialKernel(dim::Integer)
  dims = (dim, 1)
  log_variance = [0.0]
  log_length_scales = zeros(dim)
  hyperprior = (log_variance = NormalHyperprior([0.0],[1.0]), log_length_scales = NormalHyperprior(zeros(dim),ones(dim)))
  SquaredExponentialKernel(dims, log_variance, log_length_scales, hyperprior)
end

function (k::SquaredExponentialKernel)(x1::AbstractMatrix, x2::AbstractMatrix)
  (_,m) = size(x1)
  (_,n) = size(x2)
  dist = pairwise_column_difference(x1,x2) ./ exp.(k.log_length_scales)
  sq_dist = dropdims(sum(dist.^2; dims=1); dims=1)
  exp.(k.log_variance) .* exp.(-sq_dist)
end

function spectral_distribution(k::SquaredExponentialKernel, n::Integer = 1)
  Fl = eltype(k.log_variance)
  (id,_) = k.dims
  sqrt(Fl(2)) .* randn!(similar(k.log_variance,(id,1,n)))
end



# abstract type GradientEuclideanKernel{D} <: EuclideanKernel{D} end

# struct GradientSquaredExponentialKernel{D,V<:AbstractVector} <: EuclideanKernel{D}
#   log_variance      :: V
#   log_length_scales :: V
# end

# function (k::GradientSquaredExponentialKernel{D})(x1::AbstractMatrix, dx2::AbstractMatrix; dims::UnitRange=1:D) where D
#   (_,m) = size(x1)
#   (_,n) = size(dx2)
#   d = length(dims)

#   dist = pairwise_column_difference(x1, dx2) ./ exp.(k.log_length_scales)
#   sq_dist = sum(dist.^2; dims=1)
#   kernel = 2 .* exp.(k.log_variance) .* exp.(-sq_dist)

#   reshape(kernel,(1,m,n)) .* reshape(view(dist, dims,:,:) ./ exp.(view(k.log_length_scales, dims)),(d,m,n))
# end

# function (k::GradientSquaredExponentialKernel{D})(x1::AbstractMatrix, dx2::AbstractArray{<:Any,3}; dims::UnitRange=1:D) where D
#   (_,m) = size(x1)
#   (_,n1,n2) = size(dx2)
#   d = length(dims)

#   dist = pairwise_column_difference(x1, reshape(dx2, (:,n1*n2))) ./ exp.(k.log_length_scales)
#   sq_dist = sum(dist.^2; dims=1)
#   kernel = 2 .* exp.(k.log_variance) .* exp.(-sq_dist)

#   reshape(kernel,(1,m,n1,n2)) .* reshape(view(dist, dims,:,:) ./ exp.(view(k.log_length_scales, dims)),(d,m,n1,n2))
# end