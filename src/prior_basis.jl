"""
    PriorBasis

An abstract basis used for sampling a prior Gaussian process, with weights.
"""
abstract type PriorBasis end

include("prior_basis/euclidean.jl")