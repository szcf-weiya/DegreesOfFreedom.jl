ENV["PLOTS_TEST"] = "true"
ENV["GKSwstype"] = "100"
using Documenter, DegreesOfFreedom

makedocs(sitename = "DegreesOfFreedom.jl Documentation")