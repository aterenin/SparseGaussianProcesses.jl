import Flux

"""
    InterDomainOperator

An abstract inter-domain operator.
"""
abstract type InterDomainOperator end

export IdentityOperator, GradientOperator, ProductKernel

"""
    IdentityOperator

The identity inter-domain operator ``\\mathcal{A}f = f``.
"""
struct IdentityOperator <: InterDomainOperator end

"""
    GradientOperator

The gradient inter-domain operator ``\\mathcal{A}f = \\nabla f``.
"""
struct GradientOperator <: InterDomainOperator end

*(k::Kernel, op::IdentityOperator) = k
*(op::IdentityOperator, k::Kernel) = k

struct LeftGradientKernel{K<:CovarianceKernel} <: CrossCovarianceKernel
    parent::K
end
Flux.@functor LeftGradientKernel
*(op::GradientOperator, k::Kernel) = LeftGradientKernel(k)

struct RightGradientKernel{K<:CovarianceKernel} <: CrossCovarianceKernel
    parent::K
end
Flux.@functor RightGradientKernel
*(k::Kernel, op::GradientOperator) = RightGradientKernel(k)

struct GradientKernel{K<:CovarianceKernel} <: CovarianceKernel
    parent::K
end
Flux.@functor GradientKernel
*(k::LeftGradientKernel{<:CovarianceKernel}, op::GradientOperator) = GradientKernel(k.parent)
*(op::GradientOperator, k::RightGradientKernel{<:CovarianceKernel}) = GradientKernel(k.parent)





struct ProductKernel{K1<:Kernel,K2<:Kernel} <: EuclideanKernel
    dims              :: Tuple{Integer, Integer}
    kernel_one::K1
    dims_one::UnitRange{Int}
    kernel_two::K2
    dims_two::UnitRange{Int}
end
Flux.@functor ProductKernel

function ProductKernel(k1::CovarianceKernel,k2::CovarianceKernel)
    (id1,od2) = k1.dims
    (id2,od2) = k2.dims
    d1 = 1:id1
    d2 = id1 .+ (1:id2)
    ProductKernel((id1+id2,1), k1, d1, k2, d2)
end

function (k::ProductKernel)(x1::AbstractMatrix, x2::AbstractMatrix)
    @views K1 = k.kernel_one(x1[k.dims_one,:],x2[k.dims_one,:])
    @views K2 = k.kernel_two(x1[k.dims_two,:],x2[k.dims_two,:])
    K1 .* K2
end

function spectral_distribution(k::ProductKernel, n::Integer = 1)
    vcat(spectral_distribution(k.kernel_one, n), spectral_distribution(k.kernel_two, n))
end

function spectral_weights(k::ProductKernel, frequency::AbstractArray{<:Any,3})
    @views freq_one = frequency[k.dims_one,:,:]
    @views freq_two = frequency[k.dims_two,:,:]
    spectral_weights(k.kernel_one, freq_one) .* spectral_weights(k.kernel_two, freq_two)
end

function Base.getproperty(k::ProductKernel, s::Symbol)
    if s == :log_length_scales
        vcat(k.kernel_one.log_length_scales, k.kernel_two.log_length_scales)
    elseif s == :log_variance
        k.kernel_one.log_variance .+ k.kernel_two.log_variance
    else
        getfield(k,s)
    end
end

function hyperprior_logpdf(k::ProductKernel)
    hyperprior_logpdf(k.kernel_one) .+ hyperprior_logpdf(k.kernel_two)
end

# function (k::GradientKernel{<:ProductKernel})(x1::AbstractMatrix, x2::AbstractMatrix)
#     g = GradientOperator()
#     @views K1 = (g*k.parent.kernel_one*g)(x1[k.parent.dims_one,:],x2[k.parent.dims_one,:])
#     @views K2 = (g*k.parent.kernel_two*g)(x1[k.parent.dims_two,:],x2[k.parent.dims_two,:])
#     a1 = reshape(K1,)
#     a2 = ???
#     # reshape(vcat(a1, a2; dims=1), ())
# end