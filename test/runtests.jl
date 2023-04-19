using Test
using DegreesOfFreedom

@testset "lasso (iterative ridge)" begin
    tol = 1e-2
    n = 100
    p = 20
    p1 = 3
    @test cvlasso_vs_iter_ridge(n, p, p1, design = "ortho") < tol
    @test cvlasso_vs_iter_ridge(n, p, p1, design = "northo") < tol
    demo_lasso(n, p)
    @test isfile("/tmp/demo_iter_ridge_n$(n)_p$(p).pdf")
    demo_lasso_df(n, p)
    @test isfile("/tmp/demo_iter_ridge_df_n$(n)_p$(p).pdf")
end

@testset "tree" begin
    res = df_regtree(ps = [1, 2], maxd = 2)
    @test res[1, 1] < res[2, 1] < res[3, 1]
    @test res[4, 1] < res[5, 1] < res[6, 1]
end

@testset "MARS" begin
    mars_experiment_mse(ps = [2])
    @test isfile("/tmp/mars_sim.pdf")
    mars_experiment_df_vs_df(ps = [2], maxnk = 3)
    @test isfile("/tmp/df-vs-df-n100-p2-maxnk3-d2.pdf")
    @test isfile("/tmp/df-vs-df-n100-p2-maxnk3-d1.pdf")
end