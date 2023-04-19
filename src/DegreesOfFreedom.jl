module DegreesOfFreedom

include("utils.jl")
include("lasso.jl")
include("tree.jl")
include("mars.jl")

export gen_data,
       cvlasso_vs_iter_ridge,
       demo_lasso,
       demo_lasso_df,
       df_regtree,
       mars_experiment_mse,
       mars_experiment_df_vs_df

end # module DegreesOfFreedom
