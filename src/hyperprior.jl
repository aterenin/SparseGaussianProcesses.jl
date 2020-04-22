import Flux

export NormalHyperprior

abstract type Hyperprior end

struct NormalHyperprior{V<:AbstractVector} <: Hyperprior
  mean   :: V
  stddev :: V
end

Flux.trainable(k::NormalHyperprior) = ()
Flux.@functor NormalHyperprior

function (hp::NormalHyperprior)(x::AbstractVector)
  sum(((x .- hp.mean) ./ hp.stddev).^2; dims=1) ./ 2
end