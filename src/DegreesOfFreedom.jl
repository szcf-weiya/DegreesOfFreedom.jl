module DegreesOfFreedom

include("utils.jl")
include("lasso.jl")
include("tree.jl")
include("mars.jl")
include("splines.jl")
include("subset.jl")

export gen_data,
       cvlasso_vs_iter_ridge,
       demo_lasso,
       demo_lasso_df,
       df_regtree,
       run_experiment_tree,
       mars_experiment_mse,
       mars_experiment_df_vs_df,
       df_splines,
       run_experiment_splines,
       run_experiment_lasso_vs_subset

end # module DegreesOfFreedom
