module SparseGaussianProcesses

using Requires

# include("transform.jl")
include("hyperprior.jl")
include("kernel.jl")
include("prior_basis.jl")
include("inducing.jl")
include("gp.jl")
include("loss.jl")

include("utils.jl")
include("gpu.jl")

function __init__()
  @require Plots="91a5bcdd-55d7-5caf-9e0b-520d859cae80" include("plot.jl")
end

end # module