# using RCall # loaded in splines.jl
file = joinpath(@__DIR__, "cpr_lasso_subset.R")

"""
    run_experiment_lasso_vs_subset()

Run the experiment for comparing the degrees of freedom of lasso and best subset, whose results are saved into the figure shown in the paper.
"""
function run_experiment_lasso_vs_subset(; folder = "/tmp", #"../res/df"
                                        )
    R"source($file)"
    res = R"lasso_vs_subset()"
    n_lasso = rcopy(R"$res$n.lasso")
    n_subset = rcopy(R"$res$n.subset")
    df_lasso = rcopy(R"$res$df.lasso")
    df_subset = rcopy(R"$res$df.subset")
    plot(vcat(0, n_lasso), vcat(0, df_lasso), xlab = "Average # of nonzero coefficients",
            ylab = "DoF", markershape = :star5, xlims = [0, 10.5], ylims = [0, 10.5], label = "Lasso", aspect_ratio = :equal, legend=:bottomright, size = (500, 500))
            # with pgfplotx, only aspect_ratio not work, need to add `size`
    plot!(vcat(0, n_subset), vcat(0, df_subset), markershape = :circle, label = "Best Subset")
    Plots.abline!(1, 0, label="", style=:dash)
    savefig(joinpath(folder, "df_lasso_subset.pdf"))
end

