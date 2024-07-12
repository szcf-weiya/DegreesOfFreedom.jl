var documenterSearchIndex = {"docs":
[{"location":"api/#API","page":"API","title":"API","text":"","category":"section"},{"location":"api/","page":"API","title":"API","text":"Modules = [DegreesOfFreedom]\nOrder = [:type, :function]","category":"page"},{"location":"api/#DegreesOfFreedom.anim_plot-Tuple{Any, Any}","page":"API","title":"DegreesOfFreedom.anim_plot","text":"anim_plot(βs, βlasso)\n\nCompare the lasso solution from glmnet and the iterative ridge via an animated plot.\n\n\n\n\n\n","category":"method"},{"location":"api/#DegreesOfFreedom.calc_df_mars-Tuple{}","page":"API","title":"DegreesOfFreedom.calc_df_mars","text":"calc_df_mars(;n = 100, p = 10, N = 100, nk = 5, d = 1, penalty = d+1, tol = 1e-6, seedx = rand(UInt), seedy = rand(UInt))\n\nCalculate the degrees of freedom of MARS, and extract the nominal degrees of freedom used in earth::earth.\n\n\n\n\n\n","category":"method"},{"location":"api/#DegreesOfFreedom.cvlasso_vs_iter_ridge","page":"API","title":"DegreesOfFreedom.cvlasso_vs_iter_ridge","text":"cvlasso_vs_iter_ridge(n, p)\n\nCompare lasso with LOOCV and iterative ridge.\n\n\n\n\n\n","category":"function"},{"location":"api/#DegreesOfFreedom.demo_lasso","page":"API","title":"DegreesOfFreedom.demo_lasso","text":"demo_lasso(n, p, p1)\n\nDemo for Lasso fitted by iterative ridge regressions.\n\n\n\n\n\n","category":"function"},{"location":"api/#DegreesOfFreedom.demo_lasso_df","page":"API","title":"DegreesOfFreedom.demo_lasso_df","text":"demo_lasso_df(n, p)\n\nDemo for degrees of freedom of lasso via the iterative ridge regression.\n\n\n\n\n\n","category":"function"},{"location":"api/#DegreesOfFreedom.df_regtree-Tuple{}","page":"API","title":"DegreesOfFreedom.df_regtree","text":"df_regtree(; ps = [1, 5, 10], maxd = 4)\n\nExperiment for degrees of freedom for regression trees with number of features ps and maximum depth maxd.\n\n\n\n\n\n","category":"method"},{"location":"api/#DegreesOfFreedom.df_splines-Tuple{}","page":"API","title":"DegreesOfFreedom.df_splines","text":"df_splines(; Js = [5, 10, 15], λs = [0.001, 0.01, 0.1], n = 20, nrep = 10, nMC = 100)\n\nCalculate the empirical degrees of freedom of four splines:\n\ncubic splines with number of basis functions Js\nsmoothing splines with tuning parameter λs\nsample size n\nnumber of repetition nrep\nnumber of Monte Carlo samples nMC\n\n\n\n\n\n","category":"method"},{"location":"api/#DegreesOfFreedom.gen_data","page":"API","title":"DegreesOfFreedom.gen_data","text":"gen_data(n, p, p1)\n\nGenerate simulation data: \n\nX of size nxp \nβ of size p, where only the first p1 elements are signal.\ny of size p: y = Xβ + ε\n\n\n\n\n\n","category":"function"},{"location":"api/#DegreesOfFreedom.gen_data_mars","page":"API","title":"DegreesOfFreedom.gen_data_mars","text":"gen_data_mars(N = 100, p = 2)\n\nGenerate N observations from the tensor-product example with p predictors in Section 9.4.2 of Hastie et al. (2009) (The ESL book).\n\n\n\n\n\n","category":"function"},{"location":"api/#DegreesOfFreedom.iter_ridge-Tuple{AbstractMatrix, AbstractVector, Real}","page":"API","title":"DegreesOfFreedom.iter_ridge","text":"iter_ridge(X::AbstractMatrix, y::AbstractVector, λ::Real)\n\nConduct iterative ridge regression for y on X with smoothness penalty parameter λ.\n\n\n\n\n\n","category":"method"},{"location":"api/#DegreesOfFreedom.mars_experiment_df_vs_df-Tuple{}","page":"API","title":"DegreesOfFreedom.mars_experiment_df_vs_df","text":"mars_experiment_df_vs_df(; ps = [1, 10, 50], folder = \"/tmp\", maxnk = 100)\n\nCompare the nominal df and actual df for MARS for different number of predictors ps.\n\n\n\n\n\n","category":"method"},{"location":"api/#DegreesOfFreedom.mars_experiment_mse-Tuple{}","page":"API","title":"DegreesOfFreedom.mars_experiment_mse","text":"mars_experiment_mse(; folder = \"/tmp\", ps = [2, 10, 20, 30, 40, 50, 60], with_cv = false)\n\nRun MARS experiments with default MARS, and MARS with corrected penalty factor, and if with_cv, MARS with corrected penalty factor by 10-fold CV.\n\n\n\n\n\n","category":"method"},{"location":"api/#DegreesOfFreedom.mse_mars-Tuple{}","page":"API","title":"DegreesOfFreedom.mse_mars","text":"mse_mars(; d = 1, N = 200, p = 2, penalty = d+1, nk = 50, with_cv = false)\n\nCalculate the proportion of MSE decrease of the default MARS and the MARS with corrected df. If with_cv, the corrected df by cross-validation is also considered. \n\nN and p: the dimension for the data generating model\n\n\n\n\n\n","category":"method"},{"location":"api/#DegreesOfFreedom.run_experiment_lasso_vs_subset-Tuple{}","page":"API","title":"DegreesOfFreedom.run_experiment_lasso_vs_subset","text":"run_experiment_lasso_vs_subset()\n\nRun the experiment for comparing the degrees of freedom of lasso and best subset, whose results are saved into the figure shown in the paper.\n\n\n\n\n\n","category":"method"},{"location":"api/#DegreesOfFreedom.run_experiment_splines-Tuple{}","page":"API","title":"DegreesOfFreedom.run_experiment_splines","text":"run_experiment_splines()\n\nRun the experiment of splines, whose results will be saved into a .sil file (can be later loaded via deserialize) and a .tex file (the table displayed in the paper).\n\nrun_experiment_splines(folder = \"/home/weiya/Overleaf/paperDoF/res/df\")\n\n\n\n\n\n","category":"method"},{"location":"api/#DegreesOfFreedom.run_experiment_tree-Tuple{}","page":"API","title":"DegreesOfFreedom.run_experiment_tree","text":"run_experiment_tree()\n\nRun the experiment for regression tree, whose results are saved into a .sil file (can be later loaded via deserialize) and a .tex file (the table in the paper).\n\n\n\n\n\n","category":"method"},{"location":"api/#DegreesOfFreedom.save_plots-Tuple{Array}","page":"API","title":"DegreesOfFreedom.save_plots","text":"save_plots(ps; output)\n\nSave multi-images into a pdf file, if output is unspecified (default), the resulting file is /tmp/all.pdf. See also: https://github.com/szcf-weiya/Xfunc.jl/blob/master/src/plot.jl\n\n\n\n\n\n","category":"method"},{"location":"api/#DegreesOfFreedom.vary_p","page":"API","title":"DegreesOfFreedom.vary_p","text":"vary_p(d = 1, ps = [2, 10, 20, 30, 40, 50, 60]; with_cv = false, nrep = 10, nMC = 100)\n\nRun MARS experiments with different number of predictors ps for degree d. \n\nnrep: number of replications\nnMC: number of Monte Carlo samples for calculating df\nwith_cv: whether include the comparisons with the cross-validation method\n\n\n\n\n\n","category":"function"},{"location":"splines/#Splines","page":"Splines","title":"Splines","text":"","category":"section"},{"location":"splines/","page":"Splines","title":"Splines","text":"note: Note\nFor simple demonstration and fast auto-generation via GitHub Action, here we do not run the complete experiment presented in the paper. ","category":"page"},{"location":"splines/","page":"Splines","title":"Splines","text":"using DegreesOfFreedom\ndf_splines(n = 20, nrep = 2, nMC = 10)","category":"page"},{"location":"splines/","page":"Splines","title":"Splines","text":"where ","category":"page"},{"location":"splines/","page":"Splines","title":"Splines","text":"n is the sample size\nnMC is the number of Monte Carlo samples to estimate the degrees of freedom\nnrep is the number of repetitions. ","category":"page"},{"location":"splines/","page":"Splines","title":"Splines","text":"These three columns are estimated df, standard deviation of df and theoretical df, respectively.","category":"page"},{"location":"mars/#Multivariate-Adaptive-Regression-Splines-(MARS)","page":"Multivariate Adaptive Regression Splines (MARS)","title":"Multivariate Adaptive Regression Splines (MARS)","text":"","category":"section"},{"location":"mars/","page":"Multivariate Adaptive Regression Splines (MARS)","title":"Multivariate Adaptive Regression Splines (MARS)","text":"note: Note\nFor simple demonstration and fast auto-generation via GitHub Action, here we do not run the complete experiment presented in the paper. ","category":"page"},{"location":"mars/","page":"Multivariate Adaptive Regression Splines (MARS)","title":"Multivariate Adaptive Regression Splines (MARS)","text":"using DegreesOfFreedom\nmars_experiment_mse(ps = [20, 40], folder = \".\", with_cv = true)","category":"page"},{"location":"mars/","page":"Multivariate Adaptive Regression Splines (MARS)","title":"Multivariate Adaptive Regression Splines (MARS)","text":"<style>\nembed {\n    height: 300px !important;\n}\n</style>\n\n<embed src=\"../mars_sim.pdf\" width=\"800px\" height=\"2100px\" />","category":"page"},{"location":"subset/#Best-subset-vs-Lasso","page":"Best subset vs Lasso","title":"Best subset vs Lasso","text":"","category":"section"},{"location":"subset/","page":"Best subset vs Lasso","title":"Best subset vs Lasso","text":"Reproduce the comparisons of degrees of freedom between the best subset and Lasso.","category":"page"},{"location":"subset/","page":"Best subset vs Lasso","title":"Best subset vs Lasso","text":"using DegreesOfFreedom\nrun_experiment_lasso_vs_subset(folder = \".\", ylim = 12.5)","category":"page"},{"location":"subset/","page":"Best subset vs Lasso","title":"Best subset vs Lasso","text":"<style>\nembed {\n    height: 800px !important;\n}\n</style>\n\n<embed src=\"../df_lasso_subset.pdf\" width=\"800px\" height=\"2100px\" />","category":"page"},{"location":"tree/#Regression-Tree","page":"Regression Tree","title":"Regression Tree","text":"","category":"section"},{"location":"tree/","page":"Regression Tree","title":"Regression Tree","text":"note: Note\nFor simple demonstration and fast auto-generation via GitHub Action, here we do not run the complete experiment presented in the paper. ","category":"page"},{"location":"tree/","page":"Regression Tree","title":"Regression Tree","text":"using DegreesOfFreedom\ndf_regtree(ps = [1, 2], maxd = 2)","category":"page"},{"location":"tree/","page":"Regression Tree","title":"Regression Tree","text":"run_experiment_tree(nrep = 1)","category":"page"},{"location":"#DegreesOfFreedom.jl-Documentation","page":"DegreesOfFreedom.jl Documentation","title":"DegreesOfFreedom.jl Documentation","text":"","category":"section"},{"location":"","page":"DegreesOfFreedom.jl Documentation","title":"DegreesOfFreedom.jl Documentation","text":"DegreesOfFreedom.jl is a Julia package for ","category":"page"},{"location":"","page":"DegreesOfFreedom.jl Documentation","title":"DegreesOfFreedom.jl Documentation","text":"Lijun Wang, Hongyu Zhao and Xiaodan Fan (2023). Degrees of Freedom: Search Cost and Self-consistency. Manuscript.","category":"page"},{"location":"","page":"DegreesOfFreedom.jl Documentation","title":"DegreesOfFreedom.jl Documentation","text":"The documentation https://hohoweiya.xyz/DegreesOfFreedom.jl/stable/ elaborates on the degrees of freedom of various popular machine learning methods, including Lasso, Best Subset, Regression Tree, Splines and MARS.","category":"page"},{"location":"","page":"DegreesOfFreedom.jl Documentation","title":"DegreesOfFreedom.jl Documentation","text":"tip: Installation\nDegreesOfFreedom.jl is available at the General Registry, so you can easily install the package in the Julia session after typing ],julia> ]\n(@v1.8) pkg> add DegreesOfFreedom","category":"page"},{"location":"lasso/#Lasso","page":"Lasso","title":"Lasso","text":"","category":"section"},{"location":"lasso/","page":"Lasso","title":"Lasso","text":"note: Note\nFor simple demonstration and fast auto-generation via GitHub Action, here we do not run the complete experiment presented in the paper. ","category":"page"},{"location":"lasso/","page":"Lasso","title":"Lasso","text":"using DegreesOfFreedom\nn = 100\np = 20\ndemo_lasso(n, p, folder = \".\")","category":"page"},{"location":"lasso/","page":"Lasso","title":"Lasso","text":"<style>\nembed {\n    height: 2100px !important;\n}\n</style>\n\n<embed src=\"../demo_iter_ridge_n100_p20.pdf\" width=\"800px\" height=\"2100px\" />","category":"page"}]
}