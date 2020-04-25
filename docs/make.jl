using Documenter, SparseGaussianProcesses

makedocs(
  sitename = "SparseGaussianProcesses.jl", 
  modules = [SparseGaussianProcesses],
  pages = [
    "Home"=>"index.md",
    "API"=>"api.md"
  ])

deploydocs(repo = "github.com/aterenin/SparseGaussianProcesses.jl.git")