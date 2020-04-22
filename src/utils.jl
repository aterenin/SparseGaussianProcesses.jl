using Random: randperm
import Flux

export make_minibatches

function make_minibatches(x::AbstractMatrix, y::AbstractVecOrMat, z::Union{AbstractVecOrMat,Nothing} = nothing; n::Integer = 128)
  N = size(x,2)
  batch_idxs = Iterators.partition(randperm(N), n)
  if z isa Nothing
    [(x[:,i], y isa Matrix ? y[:,i] : y[i]) for i in batch_idxs]
  else 
    [(x[:,i], y isa Matrix ? y[:,i] : y[i], z isa Matrix ? z[:,i] : z[i]) for i in batch_idxs]
  end
end