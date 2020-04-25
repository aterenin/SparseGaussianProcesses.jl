using LinearAlgebra

export loss

"""
    prior_KL(gp::SparseGaussianProcess{<:Any,<:Any,<:Any,<:Any,<:MarginalInducingPoints})

Computes the prior Kullback-Leibler divergence for a Gaussian process 
with marginal inducing points, given by the expression

``KL(q(\\boldsymbol{u}) \\mathbin{||} p(\\boldsymbol{u})) = \\frac{1}{2} \\left( \\ln\\frac{|\\mathbf{K}_{\\boldsymbol{z}\\boldsymbol{z}}|}{|\\mathbf\\Sigma|} + \\operatorname{tr}(\\mathbf{K}_{\\boldsymbol{z}\\boldsymbol{z}}^{-1}\\mathbf\\Sigma) + \\boldsymbol\\mu^T \\mathbf{K}_{\\boldsymbol{z}\\boldsymbol{z}} \\boldsymbol\\mu \\right)``

where the mean is re-parameterized according to ``\\boldsymbol\\mu = \\mathbb{E}( (\\mathbf{K}_{\\boldsymbol{z}\\boldsymbol{z}} + \\xi\\mathbf{I})^{-1} \\boldsymbol{u} )``.
"""
function prior_KL(gp::SparseGaussianProcess{<:Any,<:Any,<:Any,<:Any,<:MarginalInducingPoints})
  (k,B) = (gp.kernel, gp.inter_domain_operator)
  (z,mu,U,V,QC) = gp.inducing_points()
  K = (B*k*B)(z,z)
  Q = K + V'*V
  UQ = QC.U
  logdet_term = 2 .* sum(log.(diag(UQ)) .- log.(diag(U)); dims=1)
  invUQ = inv(UQ)
  trace_term = sum((invUQ * invUQ') .* (U' * U); dims=(1,2))
  reparameterized_quadratic_form_term = sum(mu .* (Q * mu); dims=1)
  (logdet_term .- length(mu) .+ trace_term .+ reparameterized_quadratic_form_term) ./ 2
end

"""
    loss(gp::GaussianProcess, x::AbstractMatrix, y::AbstractMatrix; n_data::Int = size(x,ndims(x)))

Computes the Kullback-Leibler divergence of the variational family
from the posterior Gaussian process, up to an additive constant.
Minimizing this function trains the Gaussian process.
"""
function loss(gp::GaussianProcess, x::AbstractMatrix, y::AbstractMatrix; n_data::Int = size(x,ndims(x)))
  # sample the GP
  # rand!(gp.prior_basis)
  rand!(gp)

  # KL term
  kl = prior_KL(gp)

  # likelihood term
  n_samples = num_samples(gp)
  n_batch = size(x,ndims(x))
  f = gp(x)
  c = n_data ./ (n_batch .* n_samples .* 2)
  l = n_data .* gp.log_error .+ c .* sum(((y .- f) ./ exp.(gp.log_error)).^2; dims=(1,2))

  # regularizer term for hyperparameters
  r = gp.hyperprior.log_error(gp.log_error) .+ hyperprior_logpdf(gp.kernel)

  kl .+ l .+ r
end