using Test
using ProgressMeter
using LinearAlgebra
using Statistics
using SparseGaussianProcesses
# Random.seed!(0)

onfail(f, _::Test.Pass) = nothing
onfail(f, _::Tuple{Test.Fail,<:Any}) = f()

@testset "SparseGaussianProcesses" begin

  @testset "hyperprior" begin
    mean = [0.0,1.0]
    stddev = [1.0,1.0]
    hp = NormalHyperprior(mean, stddev)

    @test hp(mean) isa AbstractVector
    @test isapprox.(hp(mean), 0) |> all
    @test (hp(mean .+ 1) .> 0) |> all
  end

  @testset "kernel" begin
    for d in 1:2
      k = SquaredExponentialKernel(d)
      x = randn(d, 1)
      y = randn(d, 2)
      z = randn(d, 3)

      @test k.dims == (d,1)
      @test k(x,y) isa AbstractMatrix
      @test size(k(x,x)) == (1,1)
      @test (k(x,x) .== 1.0) |> all
      @test size(k(y,z)) == (2,3)
      @test (k(y,y)[1:1,1:1] .> k(y,y)[1:1,2:2]) |> all
      @test (k(y,y)[2:2,2:2] .> k(y,y)[2:2,1:1]) |> all
      @test isapprox.(k(z,z)', k(z,z)) |> all
      @test isapprox.(k(y .+ 1,y), [exp(-sqrt(d)^2) exp(-sum((y[:,2] .- (y[:,1] .+ 1)).^2)); 
                                     exp(-sum(((y[:,2] .+ 1) .- y[:,1]).^2)) exp(-sqrt(d)^2)]) |> all
    end
  end

  @testset "randomfeature" begin
    first_test = true
    for id in 1:2
      for od in 1:1
        scales = [0.1,1.,5.]
        for var in scales
          for ls in Iterators.product([scales for i in 1:id]...)
            l = 4096
            xd = 16
            s = 128
            t = 100

            k = SquaredExponentialKernel(id)
            k.log_variance .= log.(var)
            k.log_length_scales .= log.(ls)
            rf = EuclideanRandomFeatures(k, l)
            op = IdentityOperator()

            x = randn(id, xd); # sort!(x; dims=2)
            rf.weights = randn(l,s)
            K = zeros(xd, xd)

            @test size(rf(x,k,op)) == (1,xd,s)

            first_test ? first_test = false : begin @test_skip true; @info "Skipping slow RFF test"; continue end

            @showprogress for i in 1:t
              rf.weights = randn(l,s)
              out = rf(x,k,op)
              K .+= cov(reshape(out, (xd,s))')
            end
            K ./= t
            
            onfail(@test ((abs.(k(x,x) .- K) ./ var) .< 0.075) |> all) do 
              @show od id var ls l xd s t;
              display(x); println("\n")
              display(k(x,x)); println("\n")
              display(K); println("\n")
              display(abs.(k(x,x) .- K) ./ var); println("\n")
              @show maximum(abs.(k(x,x) .- K) ./ var)
            end
          end
        end
      end
    end
  end

  @testset "inducing" begin
    for id in 1:2
      for od in 1:1
        m=10
        k = SquaredExponentialKernel(id)
        ip = MarginalInducingPoints(k, m)

        (z,mu,U,V) = ip()

        @test length(mu) == od*m
        @test size(U) == (m,m)
        @test size(V) == (m,m)

        @test isapprox.(mu, 0) |> all
        @test isapprox(U' * U, k(z,z) + V'*V)
        @test isapprox(V' * V, (0.00001*I(m)))
        
        x = randn(od, m)
        ip(x,k)

        (z,mu,U,V) = ip()

        @test size(z) == size(x)
        @test length(mu) == od*m
        @test size(U) == (m,m)
        @test size(V) == (m,m)

        @test isapprox(z, x)
        @test isapprox.(mu, 0) |> all
        @test isapprox(U' * U, k(z,z) + V'*V)
        @test isapprox(V' * V, (0.00001*I(m)))

        x = randn(od, m+1)
        ip(x,k)

        (z,mu,U,V) = ip()

        @test size(z) == size(x)
        @test length(mu) == od*(m+1)
        @test size(U) == (m+1,m+1)
        @test size(V) == (m+1,m+1)

        @test isapprox(z, x)
        @test isapprox.(mu, 0) |> all
        @test isapprox(U' * U, k(z,z) + V'*V)
        @test isapprox(V' * V, (0.00001*I(m+1)))
      end
    end
  end

  @testset "gp" begin
    for id in 1:1
      for od in 1:1
        scales = [0.5,1.,1.5]
        for var in scales
          for ls in Iterators.product([scales for i in 1:id]...)
            l = 4096
            zd = 8
            xd = 128
            ns = 16384

            k = SquaredExponentialKernel(id)
            k.log_variance .= log.(var)
            k.log_length_scales .= log.(ls)
            gp = SparseGaussianProcess(k)
            gp.prior_basis = EuclideanRandomFeatures(k, l)
            gp.inducing_points = MarginalInducingPoints(k, zd)

            x = 2 .* randn(id,xd); # sort!(x; dims=2)
            z = randn(id,zd)
            u = mapslices(x -> sum(sin.(x)), z; dims=1)
            gp.inducing_points(z,k)
            gp.inducing_points.log_jitter .= log.(0.01)./2
            (_,mu,U,V) = gp.inducing_points()
            W = V' * V

            @test size(gp(z)) == (od,zd,1)

            Q = cholesky(k(z,z) + W)
            mu .= Q \ vec(u)
            K_zz = k(z,z) - k(z,z) * (Q \ k(z,z))
            LinearAlgebra.copytri!(K_zz,'U')
            gp.inducing_points.covariance_triangle .= 0
            gp.inducing_points.covariance_triangle[diagind(gp.inducing_points.covariance_triangle)] .= log(0)

            mu_x = k(x,z) * (Q \ vec(u))
            K_xx = k(x,x) - k(x,z) * (Q \ k(z,x))

            rand!(gp; num_samples = ns)
            f = gp(x)[1,:,:]
            mu_f = mean(f; dims=2)
            K_ff = cov(f')

            onfail(@test (abs.(mu_f .- mu_x) .< 0.025) |> all) do 
              @show od id var ls zd z u;
              display(mu_f'); println("\n")
              display(mu_x'); println("\n")
              @show maximum(abs.(mu_f .- mu_x))
            end

            onfail(@test ((abs.(K_xx .- K_ff) ./ var) .< 0.075) |> all) do 
              @show od id var ls zd z u;
              display(x); println("\n")
              display(K_xx); println("\n")
              display(K_ff); println("\n")
              display(abs.(K_xx .- K_ff) ./ var); println("\n")
              @show maximum(abs.(K_xx .- K_ff) ./ var)
            end
          end
        end
      end
    end
  end

  @testset "loss" begin
    k = SquaredExponentialKernel(2)
    gp = SparseGaussianProcess(k)

    @test (SparseGaussianProcesses.prior_KL(gp) .>= -1e-10) |> all
    @test (SparseGaussianProcesses.prior_KL(gp) .<= 1) |> all

    gp.inducing_points.mean .= 1
    @test (SparseGaussianProcesses.prior_KL(gp) .>= 1) |> all

    gp.inducing_points.mean .= 0
    gp.inducing_points.covariance_triangle .= 0
    @test (SparseGaussianProcesses.prior_KL(gp) .>= 1) |> all

    x = randn(1,128)
    y = randn(1,128)

    @test loss(gp,x,y) isa AbstractArray

    import Flux.gradient
    import Flux.params

    g = gradient(() -> loss(gp,x,y)|>sum, params(gp))
    for p in params(gp)
      @test isfinite.(g[p]) |> all
    end
  end

  @testset "gpu" begin

  end
end