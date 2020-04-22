using LinearAlgebra

export loss

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

function loss(gp::GaussianProcess, x::AbstractMatrix, y::AbstractMatrix; n_data::Integer = size(x,ndims(x)))
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