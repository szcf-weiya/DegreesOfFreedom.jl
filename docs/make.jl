ENV["PLOTS_TEST"] = "true"
ENV["GKSwstype"] = "100"
using Documenter, DegreesOfFreedom

makedocs(sitename = "DegreesOfFreedom.jl Documentation")

deploydocs(
    repo = "github.com/szcf-weiya/DegreesOfFreedom.jl.git",
)