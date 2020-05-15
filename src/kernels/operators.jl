"""
    InterDomainOperator

An abstract inter-domain operator.
"""
abstract type InterDomainOperator end

export IdentityOperator, GradientOperator

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
