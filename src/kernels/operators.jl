import Flux

"""
    InterDomainOperator

An abstract inter-domain operator.
"""
abstract type InterDomainOperator end

export IdentityOperator, GradientOperator, GradientKernel, ProductKernel

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
    dims:: Tuple{Int, Int}
    parent::K
end
Flux.@functor GradientKernel
GradientKernel(k::CovarianceKernel) = GradientKernel((k.dims[1],k.dims[1]), k)
*(k::LeftGradientKernel{<:CovarianceKernel}, op::GradientOperator) = GradientKernel(k.parent)
*(op::GradientOperator, k::RightGradientKernel{<:CovarianceKernel}) = GradientKernel(k.parent)

function spectral_weights(k::GradientKernel, frequency::AbstractArray{<:Any,3})
    spectral_weights(k.parent, frequency)
end




struct ProductKernel{K1<:Kernel,K2<:Kernel} <: EuclideanKernel
    dims:: Tuple{Int, Int}
    kernel_one::K1
    dims_one::UnitRange{Int}
    kernel_two::K2
    dims_two::UnitRange{Int}
end
Flux.trainable(k::ProductKernel) = (k.kernel_one, k.kernel_two)
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

function spectral_distribution(k::ProductKernel; num_samples::Integer)
    vcat(spectral_distribution(k.kernel_one; num_samples=num_samples), spectral_distribution(k.kernel_two; num_samples=num_samples))
end

function spectral_weights(k::ProductKernel, frequency::AbstractArray{<:Any,3})
    @views freq_one = frequency[k.dims_one,:,:]
    @views freq_two = frequency[k.dims_two,:,:]
    (outer_weights_one, inner_weights_one) = spectral_weights(k.kernel_one, freq_one)
    (outer_weights_two, inner_weights_two) = spectral_weights(k.kernel_two, freq_two)
    (outer_weights_one .* outer_weights_two, vcat(inner_weights_one, inner_weights_two))
end

function hyperprior_logpdf(k::ProductKernel)
    hyperprior_logpdf(k.kernel_one) .+ hyperprior_logpdf(k.kernel_two)
end

function (k::LeftGradientKernel{<:ProductKernel})(x1::AbstractMatrix, x2::AbstractMatrix)
    (_,n1) = size(x1)
    (_,n2) = size(x2)
    (d1, d2) = (length(k.parent.dims_one), length(k.parent.dims_two))
    g = GradientOperator()
    @views K1 = (k.parent.kernel_one)(x1[k.parent.dims_one,:],x2[k.parent.dims_one,:])
    @views K2 = (k.parent.kernel_two)(x1[k.parent.dims_two,:],x2[k.parent.dims_two,:])
    @views gK1 = (g*k.parent.kernel_one)(x1[k.parent.dims_one,:],x2[k.parent.dims_one,:])
    @views gK2 = (g*k.parent.kernel_two)(x1[k.parent.dims_two,:],x2[k.parent.dims_two,:])
    G1 = reshape(gK1, (d1,n1,n2)) .* reshape(K2, (1,n1,n2))
    G2 = reshape(K1, (1,n1,n2)) .* reshape(gK2, (d2,n1,n2))
    reshape(cat(G1, G2; dims=1), ((d1+d2)*n1, n2))
end

function (k::RightGradientKernel{<:ProductKernel})(x1::AbstractMatrix, x2::AbstractMatrix)
    (_,n1) = size(x1)
    (_,n2) = size(x2)
    (d1, d2) = (length(k.parent.dims_one), length(k.parent.dims_two))
    g = GradientOperator()
    @views K1 = (k.parent.kernel_one)(x1[k.parent.dims_one,:],x2[k.parent.dims_one,:])
    @views K2 = (k.parent.kernel_two)(x1[k.parent.dims_two,:],x2[k.parent.dims_two,:])
    @views gK1 = (k.parent.kernel_one*g)(x1[k.parent.dims_one,:],x2[k.parent.dims_one,:])
    @views gK2 = (k.parent.kernel_two*g)(x1[k.parent.dims_two,:],x2[k.parent.dims_two,:])
    G1 = reshape(gK1, (n1,d1,n2)) .* reshape(K2, (n1,1,n2))
    G2 = reshape(K1, (n1,1,n2)) .* reshape(gK2, (n1,d2,n2))
    reshape(cat(G1, G2; dims=2), (n1, (d1+d2)*n2))
end

function (k::GradientKernel{<:ProductKernel})(x1::AbstractMatrix, x2::AbstractMatrix)
    (_,n1) = size(x1)
    (_,n2) = size(x2)
    (d1, d2) = (length(k.parent.dims_one), length(k.parent.dims_two))
    g = GradientOperator()
    @views K1 = (k.parent.kernel_one)(x1[k.parent.dims_one,:],x2[k.parent.dims_one,:])
    @views K2 = (k.parent.kernel_two)(x1[k.parent.dims_two,:],x2[k.parent.dims_two,:])
    @views gK1 = (g*k.parent.kernel_one)(x1[k.parent.dims_one,:],x2[k.parent.dims_one,:])
    @views gKg1 = (g*k.parent.kernel_one*g)(x1[k.parent.dims_one,:],x2[k.parent.dims_one,:])
    @views gKg2 = (g*k.parent.kernel_two*g)(x1[k.parent.dims_two,:],x2[k.parent.dims_two,:])
    @views Kg2 = (k.parent.kernel_two*g)(x1[k.parent.dims_two,:],x2[k.parent.dims_two,:])
    G1 = reshape(gKg1, (d1,n1,d1,n2)) .* reshape(K2, (1,n1,1,n2))
    G2 = reshape(gK1, (d1,n1,1,n2)) .* reshape(Kg2, (1,n1,d2,n2))
    G3 = permutedims(G2, (3,2,1,4))
    G4 = reshape(K1, (1,n1,1,n2)) .* reshape(gKg2, (d2,n1,d2,n2))
    G = reshape(cat(cat(G1, G2; dims=3), cat(G3, G4; dims=3); dims=1), ((d1+d2)*n1, (d1+d2)*n2))
    n1 == n2 ? (G + G') ./ 2 : G
end