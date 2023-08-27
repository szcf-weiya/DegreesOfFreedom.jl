using GLMNet
using LinearAlgebra
using Distributions
using Plots
using Serialization
using FiniteDifferences

function ridge(X::AbstractMatrix, y::AbstractVector, λ::Real)
    # return inv(X' * X + λ * 1.0I) * X' * y
    return ridge(X, y, λ, 1.0I)
end

function ridge(X::AbstractMatrix, y::AbstractVector, λ::Real, Ψ::Union{AbstractMatrix, UniformScaling}, verbose = false)
    # A = inv(X' * X + λ * Ψ) * X'
    n = size(X, 1)
    B = X' * X /n + λ / n * Ψ
    cond = det(B) * det(inv(B))
    if verbose
        println("condition number = ", cond)
    end
    A = inv(X' * X /n + λ / n * Ψ) * X' / n
    H = X * A
    df = tr(H)
    β = A * y
    yhat = X * β
    return β, yhat, df, H
    # return inv(X' * X + λ * Ψ) * X' * y
end

function calc_df(λ, X, y, βlasso)
    β1 = copy(βlasso)
    β1[β1 .== 0] .= 1e-7
    n, p = size(X)
    # #############
    # approximate beta with fixed Ψ
    # βfunc(y::Vector) = inv(X' * X + n * λs[ind] * diagm(1 ./ abs.(β_lasso[:, ind]) )) * X' * y
    # βfunc(y::Vector) = inv(X' * X + λ / 2 * diagm(1 ./ abs.(β1) )) * X' * y
    # dψ = ForwardDiff.jacobian(ψfunc, y)
    # #####################    
    βfunc(y::Vector) = glmnet(X, y, standardize = false, intercept = false, lambda = [λ / (2n)]).betas[:]
    ## it might be problematic near zero
    # ψfunc(y::Vector) = 1 ./ abs.( βfunc(y))
    # dψ = jacobian(central_fdm(5, 1), ψfunc, y)[1]
    dβ = jacobian(central_fdm(5, 1), βfunc, y)[1]
    invXX = inv(X' * X + λ/2 * diagm(1 ./ abs.(β1) ))
    # dβ1 = invXX * X'
    dβ1 = inv(X' * X) * X'
    dβ2 = invXX * X'
    dβ3 = inv(X' * X + λ * 1.0I) * X'
    # dβ1[βlasso .== 0, :] .= 0
    # println("‖ dβ - dβ1 ‖ = ", norm(dβ - dβ1))
    diff_dβ = [norm(dβ1 - dβ), norm(dβ2 - dβ), norm(dβ3 - dβ)]
    res = zeros(n)
    res1 = zeros(n)
    res2 = zeros(n)
    res3 = zeros(n)
    # check components in dS/dy
    res_d = zeros(p, n)
    res_d2 = zeros(p, n)
    for i = 1:n
        # let NaN be zero (TODO:) what if NaN
        # dψi = dψ[:, i]
        dψi = - dβ[:, i] .* sign.(βlasso) * 1 ./ β1.^2 # note that 0 in βlasso will havs sign=0
        dψi[isnan.(dψi)] .= 0
        dS_yi = X * invXX * diagm(dψi) * invXX * X' # a matrix
        # take (i, j) element
        res[i] = sum(dS_yi[i, :] .* y)
        tmp1 = X[i, :]' * invXX * diagm(-sign.(βlasso) * 1 ./ β1.^2) * diagm(dβ[:, i])
        tmp2 = tmp1' .* β1
        # println(tmp2) # check components
        res_d[:, i] .= tmp2
        # res_d2[:, i] = 
        res1[i] = X[i, :]' * invXX * diagm(-sign.(βlasso) * 1 ./ β1.^2) * diagm(dβ[:, i]) * β1
        res2[i] = X[i, :]' * invXX * diagm(-sign.(βlasso) * 1 ./ β1 ) * dβ1[:, i]
        # res3[i] = X[i, :]' * invXX * diagm(-sign.(βlasso) * 1 ./ β1 ) * dβ1[:, i]
        # approximate dβ/dy
        dψi = - dβ1[:, i] .* sign.(βlasso) * 1 ./ β1.^2 # note that 0 in βlasso will havs sign=0
        dψi[isnan.(dψi)] .= 0
        dS_yi = X * invXX * diagm(dψi) * invXX * X' # a matrix
        # take (i, j) element
        # res1[i] = sum(dS_yi[i, :] .* y)

        # approximate dβ/dy by iter-ridge
        dψi = - dβ2[:, i] .* sign.(βlasso) * 1 ./ β1.^2 # note that 0 in βlasso will havs sign=0
        dψi[isnan.(dψi)] .= 0
        dS_yi = X * invXX * diagm(dψi) * invXX * X' # a matrix
        # take (i, j) element
        # res2[i] = sum(dS_yi[i, :] .* y)

        # approximate by ridge
        dψi = - dβ3[:, i] .* sign.(βlasso) * 1 ./ β1.^2 # note that 0 in βlasso will havs sign=0
        dψi[isnan.(dψi)] .= 0
        dS_yi = X * invXX * diagm(dψi) * invXX * X' # a matrix
        # take (i, j) element
        # res3[i] = sum(dS_yi[i, :] .* y)
    end
    df1 = tr(X * invXX * X')
    df2 = - λ/2 * sum(res)
    df21 = -λ/2 * sum(res1)
    df22 = -λ/2 * sum(res2)
    # df23 = -λ/2 * sum(res3)
    df23 = tr(invXX * diagm( sign.(βlasso) ./ β1)) * λ / 2
    # tmp_mat = invXX * diagm( 0.5 ./ abs.(β1))
    df3 = sum(βlasso .!= 0) # nonzero coefficients
    return df1, df2, df3, df21, df22, df23, diff_dβ#, res_d
end

function weight_matrix(β::AbstractVector, tol::Real)
    a = 1 ./ abs.(β)
    a[a .> 1 / tol] .= 1 / tol
    return diagm(a / 2)
end

"""
    iter_ridge(X::AbstractMatrix, y::AbstractVector, λ::Real)

Conduct iterative ridge regression for `y` on `X` with smoothness penalty parameter `λ`.
"""
function iter_ridge(X::AbstractMatrix, y::AbstractVector, λ::Real; tol = sqrt(eps()), 
                    maxiter = 10000, 
                    remove = false,
                    err_type = "norm",
                    verbose = false,
                    # β1 = ridge(X, y, λ)
                    )
    n, p = size(X)
    βs = zeros(p, 0)
    yhats = zeros(n, 0)
    dfs = Float64[]
    β1, yhat, df = ridge(X, y, λ)
    # β1 =β_lasso[:, 38]
    βs = hcat(βs, β1)
    yhats = hcat(yhats, yhat)
    append!(dfs, df)
    iter = 0
    nzeros = 0
    last_nzeros = 0
    H = nothing
    while true
        iter += 1
        if remove
            idx = abs.(βs[:, end]) .> tol
            # idx = abs.(βs[:, end]) .> eps()
            # idx = abs.(βs[:, end]) .> 0.0
            nzeros = sum(.!idx)
            β2 = zeros(p)
            if nzeros == last_nzeros
                # update Ψ
                Ψ = diagm(1 ./ abs.(βs[idx, end]))
                β2[idx], yhat, df = ridge(X[:, idx], y, λ, Ψ)    
            else
                # restart ridge
                println("restart ridge regression, nzeros = $nzeros")
                β2[idx], yhat, df = ridge(X[:, idx], y, λ)
            end
            last_nzeros = nzeros
        else
            Ψ = weight_matrix(βs[:, end], tol)
            β2, yhat, df, H = ridge(X, y, λ, Ψ)    
        end
        if err_type == "norm"
            err = norm(βs[:, end] - β2)
        else
            err = maximum(abs.(βs[:, end] - β2))
        end
        if verbose
            println("iter = $iter, err = $err")
        end
        if (err < tol) | (iter > maxiter)
            break
        end
        βs = hcat(βs, β2)
        yhats = hcat(yhats, yhat)
        append!(dfs, df)    
    end
    return βs, yhats, dfs, H
end

"""
    gen_data(n, p, p1)

Generate simulation data: 

- `X` of size `nxp` 
- `β` of size `p`, where only the first `p1` elements are signal.
- `y` of size `p`: `y = Xβ + ε`
"""
function gen_data(n = 100, p = 20, p1 = 3; standardize_y = false, design = "ortho")
    if design == "ortho"
        X = randn(n, p)
    else
        Σ = zeros(p, p)
        for i = 1:p
            for j = i:p
                Σ[i, j] = 0.7^abs(j-i)
                Σ[j, i] = Σ[i, j]
            end
        end
        X = rand(MvNormal(zeros(p), Σ), n)'
    end
    β = zeros(p)
    β[1:p1] .= 1
    y = X * β + randn(n)
    if standardize_y
        # scale y to keep agree with glmnet
        # https://cran.r-project.org/web/packages/glmnet/glmnet.pdf
        # """
        # Note also that for "gaussian", glmnet standardizes y to have unit variance (using 1/n rather than
        # 1/(n-1) formula) before computing its lambda sequence (and then unstandardizes the resulting coefficients); # if you wish to reproduce/compare results with other software, best to supply a standardized
        # y. The coefficients for any predictor variables with zero variance are set to zero for all values of
        # lambda.
        # """
        k = sqrt(var(y, corrected = false))
        return X/k, y/k, β, k
    else
        return X, y, β
    end
end

"""
    demo_lasso_df(n, p)

Demo for degrees of freedom of lasso via the iterative ridge regression.
"""
function demo_lasso_df(n = 100, p = 20, p1 = 5, folder = "/tmp")
    X, y, β = gen_data(n, p, p1, standardize_y = true, design = "ortho")
    n, p = size(X)
    res_cvlasso = glmnetcv(X, y, nfolds = n, standardize = false, intercept = false)
    λs = res_cvlasso.lambda
    nλ = length(λs)
    β_lasso = res_cvlasso.path.betas
    ind = argmin(res_cvlasso.meanloss)
    # DFS = zeros(nλ, 3)
    DFS = zeros(nλ, 9)
    for ind = 1:nλ
        tmp = calc_df(λs[ind]*2n, X, y, β_lasso[:, ind])
        DFS[ind, 1:6] .= tmp[1:6]
        DFS[ind, 7:9] .= tmp[7]
        # DFS[ind, :] .= calc_df(λs[ind]*2n, X, y, β_lasso[:, ind])
    end
    figname = joinpath(folder, "demo_iter_ridge_df_n$(n)_p$(p)")
    serialize(figname * ".sil", DFS)
    figs = Plots.Plot[]
    fig_dfs = plot(λs, DFS[:, 3], label = "#nonzeros", title = "n = $n, p = $p", ls = :dot, lw = 2, legend = :topright)
    plot!(fig_dfs, λs, DFS[:, 1], label = "trace")
    plot!(fig_dfs, λs, DFS[:, 1] + DFS[:, 2], label = "trace + δ")
    fig_diff = plot(λs, DFS[:, 1] + DFS[:, 2] - DFS[:, 3], label = "trace + δ - #nonzeros", title = "n = $n, p = $p", legend = :topright)
    push!(figs, fig_dfs)
    push!(figs, fig_diff)
    save_plots(figs, output = figname * ".pdf")
end

"""
    demo_lasso(n, p, p1)

Demo for Lasso fitted by iterative ridge regressions.
"""
function demo_lasso(n = 100, p = 20, p1 = 5; folder = "/tmp")
    @assert p1 < p "p1 should be smaller than p"
    X, y, β = gen_data(n, p, p1, standardize_y = true)
    res_cvlasso = glmnetcv(X, y, nfolds = n, standardize = false, intercept = false)
    λs = res_cvlasso.lambda
    res_lasso = glmnet(X, y, standardize = false, intercept = false, lambda = λs)
    β_lasso = res_lasso.betas
    nλ = length(λs)
    errs = zeros(nλ)
    gcvs = zeros(nλ)
    DFS = zeros(4, nλ)
    RSS = zeros(nλ)
    RSS0 = zeros(nλ)
    tol = eps()^(1/3)
    for ind = 1:nλ
        βs, yhats, dfs = iter_ridge(X, y, 2n*λs[ind], tol = tol, remove = false, err_type = "max") 
        errs[ind] = norm(βs[:, end] - β_lasso[:, ind])
        DFS[1, ind] = dfs[1]
        DFS[2, ind] = dfs[end]
        df_lasso = rank(X[:,β_lasso[:,ind] .!= 0])
        df_lasso1 = sum(abs.(βs[:, end]) .> tol)
        DFS[3, ind] = df_lasso
        DFS[4, ind] = df_lasso1
        RSS[ind] = norm(yhats[:, end] - y)^2
        RSS0[ind] = norm(yhats[:, 1] - y)^2
    end
    figname = joinpath(folder, "demo_iter_ridge_n$(n)_p$(p)")
    serialize(figname * ".sil", [X, y, β, DFS, RSS, RSS0, errs, λs, tol, res_cvlasso])
    # return X, y, β, DFS, RSS, RSS0, errs, λs, tol, res_cvlasso
    save_plots(plot_demo(X, y, β, DFS, RSS, RSS0, errs, λs, tol, res_cvlasso), output = figname * ".pdf")
end

function plot_demo(X, y, β, DFS, RSS, RSS0, errs, λs, tol, res_cvlasso)
    n, p = size(X)
    gcvs0 = RSS0 ./ (1 .- DFS[1, :] / n).^2 / n
    gcvs = RSS ./ (1 .- DFS[2, :] / n).^2 / n
    gcvs11 = RSS ./ (1 .- DFS[4, :] / n).^2 / n
    ## err plot
    fig_err = plot(λs, errs)
    ## coef plot (an example)
    ind = argmin(res_cvlasso.meanloss)
    βs, yhats, dfs = iter_ridge(X, y, 2n*λs[ind], tol = tol, remove = false, err_type = "max")
    m = size(βs, 2)
    alphas = range(0.2, 1, length = m)
    if (m == 3)
        lbls = vcat("Ridge", "IterRidge (1 step)", "IterRidge (converged)")
    else
        lbls = vcat("Ridge", "IterRidge (1 step)", "IterRidge (n step)", repeat([""], m-4), "IterRidge (converged)")
    end
    fig_βs = plot(βs, color = :red, alpha = reshape(alphas, 1, :), label = reshape(lbls, 1, :), 
                        title = "n = $n, p = $p, λ = $(round.(λs[ind]*2n, digits=3))", 
                        xlab = "i", ylab = "β[i]", legend = :topright) 
    plot!(fig_βs, β, label = "Truth", ls = :dash)
    plot!(fig_βs, coef(res_cvlasso), ls = :dot, label = "Lasso", lw = 2)
    ## gcv plot
    fig_gcv = plot(λs, gcvs, label = "GCV of IterRidge (df = tr(H))", markershape = :x, legend = :topleft, xlab = "λ", ylab = "GCV", title = "n = $n, p = $p")
    plot!(fig_gcv, λs, gcvs11, label = "GCV of IterRidge (df = #{> tol})", markershape = :vline)
    # plot!(λs, gcvs0, label = "GCV of Ridge", markershape = :star5)
    plot!(fig_gcv, λs, res_cvlasso.meanloss, label = "LOOCV of Lasso", markershape = :+)
    # plot!(λs, gcvs1, label = "GCV of IterRidge (df = #nonzeros)", markershape = :circle)
    ## df plot
    fig_df = plot(λs, DFS[2, :], label = "tr(H) of IterRidge", xlab = "λ", ylab = "DoF", title = "n = $n, p = $p", legend = :topright)
    plot!(fig_df, λs, DFS[4, :], label = "#{> tol} of IterRidge")
    plot!(fig_df, λs, DFS[3, :], label = "#nonzeros of Lasso ")
    return fig_err, fig_βs, fig_df, fig_gcv
end


function MMfunc()
    g1(β, β0) = abs(β0) + 0.5 * (β^2 - β0^2) / β0^2
    g2(β, β0) = 0.5 / abs(β0) * β^2 + abs(β0) / 2
    g1(β0) = β -> g1(β, β0)
    g2(β0) = β -> g2(β, β0)
    f(β) =  abs(β)
    βs = range(-2, 2, length = 100)
    β0 = 0.5
    plot(βs, f.(βs))
    plot!(βs, g1(β0).(βs), ls = :dash, lw = 2)
    plot!(βs, g2(β0).(βs), lw = 2)
end

"""
    anim_plot(βs, βlasso)

Compare the lasso solution from glmnet and the iterative ridge via an animated plot.
"""
function anim_plot(βs, βlasso)
    m = size(βs, 2)
    anim = nothing
    alphas = range(0.2, 1, length = m)
    for i = 1:m
        if i == 1
            anim = Animation()
            plot(βs[:, 1], legend = false, ylim = (-0.5, 1), color = :red, alpha = alphas[i])
            plot(βlasso, lw = 2)
        else
            plot!(βs[:, i], legend = false, color = :red, alpha = alphas[i])
            frame(anim)
        end
    end
    # return anim
    gif(anim, fps = 1)
    mp4(anim, fps = 1)
    # mp4(anim)
    return anim
end

"""
    cvlasso_vs_iter_ridge(n, p)

Compare lasso with LOOCV and iterative ridge.
"""
function cvlasso_vs_iter_ridge(n = 100, p = 20, p1 = 5; tol = cbrt(eps()), design = "ortho")
    X, y, β = gen_data(n, p, p1, standardize_y = true, design = design)
    res_cvlasso = glmnetcv(X, y, nfolds = n, standardize = false, intercept = false)
    β_lasso = res_cvlasso.path.betas
    ind = argmin(res_cvlasso.meanloss)
    λs = res_cvlasso.lambda
    βs, yhats, dfs = iter_ridge(X, y, 2n*λs[ind], tol = tol, remove = false, err_type = "max")
    return norm(βs[:, end] - β_lasso[:, ind]) / norm(β)
end