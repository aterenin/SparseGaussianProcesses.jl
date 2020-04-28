import Flux

export NormalHyperprior

"""
    Hyperprior

An abstract hyperprior.
"""
abstract type Hyperprior end

"""
    NormalHyperprior

An isotropic multivariate normal hyperprior, parameterized by mean and 
element-wise standard deviation.
"""
struct NormalHyperprior{V<:AbstractVector} <: Hyperprior
  mean   :: V
  stddev :: V
end

Flux.trainable(k::NormalHyperprior) = ()
Flux.@functor NormalHyperprior

"""
    (hp::NormalHyperprior)(x::AbstractVector)

Evaluates the normal hyperprior at ``x``.
"""
function (hp::NormalHyperprior)(x::AbstractVector)
  sum(((x .- hp.mean) ./ hp.stddev).^2; dims=1) ./ 2
end