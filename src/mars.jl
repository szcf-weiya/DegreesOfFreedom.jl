using RCall
using Random
using LaTeXStrings
using StatsPlots

"""
    gen_data_mars(N = 100, p = 2)

Generate `N` observations from the tensor-product example with `p` predictors in Section 9.4.2 of Hastie et al. (2009) (The ESL book).
"""
function gen_data_mars(N = 100, p = 2)
    x1 = randn(N)
    x2 = randn(N)
    if p > 2
        xo = randn(N, p - 2)
    end
    x1_plus = (x1 .> 1) .* (x1 .- 1)
    x2_plus = (x2 .> 0.8) .* (x2 .- 0.8)
    ε = randn(N)
    y0 = x1_plus + x1_plus .* x2_plus
    y = y0 + 0.12ε
    #println("SNR:", var(y0) / 0.12^2)
    x = hcat(x1, x2)
    if p > 2
        x = hcat(x, xo)
    end
    return x, y, y0
end

"""
    mse_mars(; d = 1, N = 200, p = 2, penalty = d+1, nk = 50, with_cv = false)

Calculate the proportion of MSE decrease of the default MARS and the MARS with corrected df.
If `with_cv`, the corrected df by cross-validation is also considered. 
- `N` and `p`: the dimension for the data generating model
"""
function mse_mars(; d = 1, N = 200, p = 2, penalty = d+1, nk = 50, with_cv = false)
    x, y, y0 = gen_data_mars(N, p) # in the text, it is 100, but the accuracy is not so high.
    xt, yt, y0t = gen_data_mars(1000, p)
    R"fit = earth::earth($x, $y, degree = $d, nk = $nk, thresh = 0)"
    R"fit2 = earth::earth($x, $y, degree = $d, penalty = $penalty, nk = $nk, thresh = 0)"
    yhat = rcopy(R"predict(fit, $xt)")
    yhat2 = rcopy(R"predict(fit2, $xt)")
    mse = mean((yhat - y0t).^2)
    mse2 = mean((yhat2 - y0t).^2)
    mse0 = mean((mean(yt) .- y0t).^2)
    if with_cv
        tcv = @elapsed penalty = cv_penalty(x, y, arr_penalty = 0.5:0.5:6, nfold = 10, d = d, nk = nk, plt = true)
        println("cv penalty = $penalty")
        R"fitcv = earth::earth($x, $y, degree = $d, penalty = $penalty, nk = $nk, thresh = 0)"
        yhatcv = rcopy(R"predict(fitcv, $xt)")
        msecv = mean((yhatcv - y0t).^2)
        (mse0 - mse) / mse0, (mse0 - mse2) / mse0, (mse0 - msecv) / mse0, penalty, tcv
    else
        (mse0 - mse) / mse0, (mse0 - mse2) / mse0
    end
end

function cv_penalty(x::AbstractArray{T}, y::AbstractArray{T}; nfold = 10, arr_penalty = [2, 3], d = 1, nk = 50, plt = false, one_se_error = false) where T <: AbstractFloat
    n = size(x, 1)
    folds = MonotoneSplines.div_into_folds(n, K = nfold)
    npenalty = length(arr_penalty)
    errs = zeros(npenalty)
    ses = zeros(npenalty)
    for i in 1:npenalty
        ei = zeros(nfold)
        for k in 1:nfold
            test_idx = folds[k]
            train_idx = setdiff(1:n, test_idx)        
            R"fit = earth::earth($(x[train_idx, :]), $(y[train_idx]), degree = $d, thresh = 0, penalty = $(arr_penalty[i]), nk = $nk)"
            yhat = rcopy(R"predict(fit, $(x[test_idx, :]) )")
            ei[k] = mean((yhat .- y[test_idx]).^2)
        end
        ses[i] = std(ei) / sqrt(nfold)
        errs[i] = mean(ei)
    end
    if plt
        p = size(x, 2)
        #filename = "cv_penalty-p$p-d$d-$(time_ns()).pdf"
        filename = "cv_penalty-p$p-d$d.pdf"
        savefig(plot(arr_penalty, errs, yerror = ses, legend = false, xlab = "penalty", ylab = "CV error"), "/tmp/$filename")
    end
    if one_se_error
        idx = argmin(errs)
        return arr_penalty[findfirst(errs .< errs[idx] + ses[idx])]
    else
        return arr_penalty[argmin(errs)]
    end
end

function compare_mse(; N = 200, p = 20, nrep = 5, nk = min(200, max(20, 2p+1)), d = 1, with_cv = false, nMC = 100)
    tprop = @elapsed res = sol_mars_df_and_penalty(d = d, n = N, p = p, nk = nk, N = nMC) # work for 1 temporarily
    penalty = res[3]
    @info "adaptive penalty = $penalty"
    props = [mse_mars(d = d, N = N, p = p, penalty = penalty, nk = nk, with_cv = with_cv) for i = 1:nrep]
    diff = [m[2] - m[1] for m in props]
    p1 = mean([m[1] for m in props])
    s1 = std([m[1] for m in props])
    p2 = mean([m[2] for m in props])
    s2 = std([m[2] for m in props])
    if with_cv
        pcv = mean([m[3] for m in props])
        scv = std([m[3] for m in props])
        cvpenalty = mean([m[4] for m in props])
        scvpenalty = std([m[4] for m in props])
        tcv = mean([m[5] for m in props])
        # return sum(diff .> 0), mean(diff), p1, p2
        return sum(diff .> 0), sum(diff .== 0), p1, p2, s1, s2, penalty, pcv, scv, cvpenalty, scvpenalty, tcv, tprop
    else
        return sum(diff .> 0), sum(diff .== 0), p1, p2, s1, s2, penalty
    end
end

function df_vs_df(;d = 1, maxnk = 20, p = 10, n = 100)
    df1 = zeros(maxnk, 2)
    df2 = zeros(maxnk, 2)
    for nk = 1:maxnk
        res = sol_mars_df_and_penalty(n = n, p = p, nk = nk, d = d)
        df1[nk, 1] = res[1]
        df1[nk, 2] = res[2]
        df2[nk, :] .= calc_df_mars(n = n, p = p, nk = nk, penalty = d+1, d = d)
    end
    return df1, df2
end

function plot_df_mars(d = 1, maxnk = 20, p = 10, n = 100; method = "iter", maxiter = 3)
    penalty = d + 1
    df = zeros(maxnk, 2)
    for nk = 1:maxnk
        # R"res = calc_df(nk = $nk, d = $d)"
        # df[nk, 1] = R"res$df"
        # df[nk, 2] = R"res$dfbar"
        if method == "iter"
            res = sol_mars_df_and_penalty(n = n, p = p, nk = nk, d = d)
            df[nk, 1] = res[1][end]
            df[nk, 2] = res[2][end]
        else
            df[nk, :] .= calc_df_mars(n = n, p = p, nk = nk, penalty = penalty, d = d)
        end
    end
    # return df
#    plot(1:maxnk, df, label = ["df (empirical based on Cov)" "df (approximated in MARS)"], title = "penalty = $penalty")
    plot(df[:, 1], df[:, 2], xlab = "empirical df", ylab = "df in MARS", markershape=:star5, aspect_ratio = 1, label = "", 
                title = ifelse(method == "iter", "corrected penalty (degree = $d, p = $p)", "default penalty = $penalty (degree = $d, p = $p)"), 
                xlim = [0, maximum(df)+1],
                ylim = [0, maximum(df)+1],
                # size = (800, 800)
                )
    Plots.abline!(1, 0, label="", style = :dash)
    # savefig("../res/df/df-vs-df-$method-n$n-p$p-maxnk$maxnk-d$d.pdf")
end

function sol_mars_df_and_penalty(; n = 20, p = 2, N = 100, nk = 20, d = 1, seedx = 1234, seedy = 5432)
    # init penalty
    penalty = d + 1
    iter = 0
    # based on our experiments, the number of selected terms is somewhat robust to penalty, 
    # so the loop would finish after two iterations.
    df_cov, df_app = calc_df_mars(penalty = penalty, n = n, p = p, N = N, nk = nk, d = d, seedx = seedx, seedy = seedy)
    # average # of selected terms
    avg_m = (df_app + penalty / 2) / (penalty / 2 + 1)
    # solve new penalty
    if avg_m > 1 && df_cov > avg_m
        penalty = 2(df_cov - avg_m )/(avg_m - 1)
    end
    return df_cov, df_app, penalty
end

"""
    calc_df_mars(;n = 100, p = 10, N = 100, nk = 5, d = 1, penalty = d+1, tol = 1e-6, seedx = rand(UInt), seedy = rand(UInt))

Calculate the degrees of freedom of MARS, and extract the nominal degrees of freedom used in `earth::earth`.
"""
function calc_df_mars(;n = 100, p = 10, N = 100, nk = 5, d = 1, penalty = d+1, tol = 1e-6, seedx = rand(UInt), seedy = rand(UInt))
    y = randn(MersenneTwister(seedy), n, N)
    x = randn(MersenneTwister(seedx), n, p)
    yhat = zeros(n, N)
    dfhat = zeros(N)
    for i = 1:N
        R"mars.fit = earth::earth($x, $(y[:, i]), nk = $nk, thresh = 0, pmethod = 'none', degree = $d, penalty = $penalty)"
        yhat[:, i] = rcopy(R"predict(mars.fit)")
        ns = rcopy(R"length(mars.fit$selected.terms)")
        # dfhat[i] = cm[i] + penalty * (ms[i] - 1) / 2 # exclude the intercept; divide by 2 comes from the code L1033 version 5.3.1
        dfhat[i] = ns + penalty * ((ns - 1)/2)
        # check gcv with dfhat[i]
        gcv = rcopy(R"mars.fit$rss") / n / (1 - dfhat[i] / n)^2
        gcv0 = rcopy(R"mars.fit$gcv")
        if abs(gcv0 - gcv) > tol
            @warn "earth report gcv = $gcv0, calculated gcv = $gcv"
        end
    end
    df = sum([cov(y[i, :], yhat[i, :]) for i = 1:n])
    return df, mean(dfhat)
end

"""
    vary_p(d = 1, ps = [2, 10, 20, 30, 40, 50, 60]; with_cv = false, nrep = 10, nMC = 100)

Run MARS experiments with different number of predictors `ps` for degree `d`. 

- `nrep`: number of replications
- `nMC`: number of Monte Carlo samples for calculating df
- `with_cv`: whether include the comparisons with the cross-validation method
"""
function vary_p(d = 1, ps = [2, 10, 20, 30, 40, 50, 60]; with_cv = false, nrep = 10, nMC = 100)
#    ps = [10]
    RES = zeros(length(ps), 7 + ifelse(with_cv, 6, 0))
    for (i, p) in enumerate(ps)
        RES[i, :] .= compare_mse(p = p, nrep = nrep, d = d, with_cv = with_cv, nMC = nMC)
    end
    return RES
end

"""
    mars_experiment_mse(; folder = "/tmp", ps = [2, 10, 20, 30, 40, 50, 60], with_cv = false)

Run MARS experiments with default MARS, and MARS with corrected penalty factor, and if `with_cv`, MARS with corrected penalty factor by 10-fold CV.
"""
function mars_experiment_mse(; folder = "/tmp", ps = [2, 10, 20, 30, 40, 50, 60], with_cv = false)
    RES1 = vary_p(1, ps, with_cv = with_cv)
    serialize(joinpath(folder, "mars_varyp_d1.sil"), RES1)
    RES2 = vary_p(2, ps, with_cv = with_cv)
    serialize(joinpath(folder, "mars_varyp_d2.sil"), RES2)

    N = 10
    scale_factor = 1
    p1 = plot(ps, RES1[:, 3], yerror = RES1[:, 5] ./ sqrt(N) / scale_factor, xlab = L"p", ylab = L"R^2", 
                xticks = (ps, string.(ps)), label = "default", title = "degree = 1", ls = :dot, 
                legend = :bottomleft,
                labelfontsize = 14, legendfontsize = 14, tickfontsize = 12, titlefontsize = 16
                # legend = false # shared with p2
    )
    plot!(p1, ps, RES1[:, 4], yerror = RES1[:, 6] ./ sqrt(N) / scale_factor, label = "corrected")
    if with_cv
        plot!(p1, ps, RES1[:, 8], yerror = RES1[:, 9] ./ sqrt(N) / scale_factor, label = "10-fold CV", ls = :dash)
    end
    savefig("$folder/mars_p1.pdf")
    p2 = plot(ps, RES2[:, 3], yerror = RES2[:, 5] ./ sqrt(N) / scale_factor, xlab = L"p", ylab = L"R^2", xticks = (ps, string.(ps)), label = "default", title = "degree = 2", ls = :dot,
        # ylim = [0, 1.0]
        legend = false, # share with p1
        labelfontsize = 14, legendfontsize = 14, tickfontsize = 12, titlefontsize = 16
        )
    plot!(p2, ps, RES2[:, 4], yerror = RES2[:, 6] ./ sqrt(N) / scale_factor, label = "corrected")
    if with_cv
        plot!(p2, ps, RES2[:, 8], yerror = RES2[:, 9] ./ sqrt(N) / scale_factor, label = "10-fold CV", ls = :dash)
    end
    savefig("$folder/mars_p2.pdf")
    cols = [:blue :orange]
    p3 = plot(ps, RES1[:, 7], xticks = (ps, string.(ps)), xlab = L"p", ylab = "penalty", label = "corrected (degree = 1)", color = cols[1], markershape = :star5, 
                legend=:bottomright, ylim = (0.5, 6.5),
                labelfontsize = 14, legendfontsize = 14, tickfontsize = 12, titlefontsize = 16)
    plot!(p3, ps, RES2[:, 7], label = "corrected (degree = 2)", color = cols[2], markershape = :dtriangle)
    if with_cv
        plot!(p3, ps, RES1[:, 10], yerror = RES1[:, 11] ./ sqrt(N), ls = :dash, color = cols[1], label = "10-fold CV (degree = 1)")
        plot!(p3, ps, RES2[:, 10], yerror = RES2[:, 11] ./ sqrt(N), ls = :dash, color = cols[2], label = "10-fold CV (degree = 2)")
    end
    hline!(p3, [2 3], label = ["default (degree = 1)" "default (degree = 2)"], style = :dot, color = cols)
    savefig("$folder/mars_p3.pdf")
    p11 = groupedbar(ps, RES1[:, 1:2] ./ 10, bar_position=:stack, xticks = (ps, string.(ps)), 
        label = ["corrected > default" "corrected = default"], 
        # legend = :topleft, 
        # legend = false,
        xlab = L"p", ylab = "percent")
    savefig("$folder/mars_p11.pdf")
    p21 = groupedbar(ps, RES2[:, 1:2] ./ 10, bar_position=:stack, xticks = (ps, string.(ps)), label = ["corrected > default" "corrected = default"], xlab = L"p", ylab = "percent", legend = false)
    savefig("$folder/mars_p21.pdf")
    # use `_` to occupy some space to avoid overlap of the margin
    # use the sum of width less than one 
    # although the sum of with < 1, without _, the width would be rescaled
    # so leave one columns without width
    # also, use a single `_` is enough, but do not specify width for it like `_{0.3w}`, which would be rendered as a normal symbol
    #plot(p1, p11, p2, p21, p3, layout=@layout([grid(2, 1){0.3w} _ grid(2, 1){0.3w} b{0.33w}]), size = (1300, 400)) # too small margin overlap legend
    #savefig("$folder/mars_sim.pdf")


    plot(p1, p2, p3, layout = @layout([a b c]), size = (1400, 400)) # too small margin overlap legend
    savefig("$folder/mars_sim.pdf")
end

"""
    mars_experiment_df_vs_df(; ps = [1, 10, 50], folder = "/tmp", maxnk = 100)

Compare the nominal df and actual df for MARS for different number of predictors `ps`.
"""
function mars_experiment_df_vs_df(; ps = [1, 10, 50], folder = "/tmp", maxnk = 100)
    for method in ["default", "iter"]
        for p in ps
            for d in [1, 2]
                plot_df_mars(d, maxnk, p, method = method)
            end
        end
    end
    RES = [[df_vs_df(d = d, maxnk = maxnk, p = p) for d in [1, 2]] for p in ps]
    serialize("$folder/mars_df_vs_df.sil", RES)
    for (i, p) in enumerate(ps)
        for d in [1, 2]
            r = max(maximum(RES[i][d][1]), maximum(RES[i][d][2]))
            plot(RES[i][d][1][:, 1], RES[i][d][1][:, 2], xlab = "empirical df", ylab = "df in MARS", 
                                                         markershape=:star5, aspect_ratio = 1, label = "corrected", 
                                                         title = "degree = $d, p = $p", 
                                                        xlim = [0, r],
                                                        ylim = [0, r],
                                                        size = (500, 500),
                                                        legend = :topleft
            )
            plot!(RES[i][d][2][:, 1], RES[i][d][2][:, 2], label = "default", markershape = :dtriangle)
            Plots.abline!(1, 0, label="", style = :dash)
            savefig("$folder/df-vs-df-n100-p$p-maxnk$maxnk-d$d.pdf")
        end
    end
end