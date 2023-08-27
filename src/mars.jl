using RCall
using Random
using LaTeXStrings
using StatsPlots

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
    # println("SNR:", var(y0) / 0.12^2)
    x = hcat(x1, x2)
    if p > 2
        x = hcat(x, xo)
    end
    return x, y, y0
end

function mse_mars(;d = 1, N = 200, p = 2, penalty = d+1, nk = nk)
    x, y, y0 = gen_data_mars(N, p) # in the text, it is 100, but the accuracy is not so high.
    xt, yt, y0t = gen_data_mars(1000, p)
    R"fit = earth::earth($x, $y, degree = $d, nk = $nk)"
    R"fit2 = earth::earth($x, $y, degree = $d, penalty = $penalty, nk = $nk)"
    yhat = rcopy(R"predict(fit, $xt)")
    yhat2 = rcopy(R"predict(fit2, $xt)")
    mse = mean((yhat - y0t).^2)
    mse2 = mean((yhat2 - y0t).^2)
    mse0 = mean((mean(yt) .- y0t).^2)
    (mse0 - mse) / mse0, (mse0 - mse2) / mse0
end

function compare_mse(; N = 200, p = 20, nrep = 5, maxiter = 5, nk = 100, d = 1)
    res = sol_mars_df_and_penalty(d = d, n = N, p = p, nk = nk, maxiter = maxiter) # work for 1 temporarily
    penalty = res[3][end]
    @info "adaptive penalty = $penalty"
    props = [mse_mars(d = d, N = N, p = p, penalty = penalty, nk = nk) for i = 1:nrep]
    diff = [m[2] - m[1] for m in props]
    p1 = mean([m[1] for m in props])
    s1 = std([m[1] for m in props])
    p2 = mean([m[2] for m in props])
    s2 = std([m[2] for m in props])
    # return sum(diff .> 0), mean(diff), p1, p2
    return sum(diff .> 0), sum(diff .== 0), p1, p2, s1, s2, penalty
end

function df_vs_df(;d = 1, maxnk = 20, p = 10, n = 100, maxiter = 3)
    df1 = zeros(maxnk, 2)
    df2 = zeros(maxnk, 2)
    for nk = 1:maxnk
        res = sol_mars_df_and_penalty(n = n, p = p, nk = nk, maxiter = maxiter, d = d)
        df1[nk, 1] = res[1][end]
        df1[nk, 2] = res[2][end]
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
            res = sol_mars_df_and_penalty(n = n, p = p, nk = nk, maxiter = maxiter, d = d)
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

function sol_mars_df_and_penalty(; n = 20, p = 2, N = 100, nk = 5, tol = 1e-3, maxiter = 10, trace_plot = false, d = 1, seedx = 1234, seedy = 5432)
    arr_penalty = Float64[]
    arr_dfcov = Float64[]
    arr_dfapp = Float64[]
    # init penalty
    penalty = d + 1
    iter = 0
    while true
        push!(arr_penalty, penalty)
        # based on our experiments, the number of selected terms is somewhat robust to penalty, 
        # so the loop would finish after two iterations.
        df_cov, df_app = calc_df_mars(penalty = penalty, n = n, p = p, N = N, nk = nk, d = d, seedx = seedx, seedy = seedy)
        push!(arr_dfcov, df_cov)
        push!(arr_dfapp, df_app)
        # average # of selected terms
        avg_m = (df_app + penalty / 2) / (penalty / 2 + 1)
        # solve new penalty
        if avg_m > 1 && df_cov > avg_m
            penalty = 2(df_cov - avg_m )/(avg_m - 1)
        end
        if length(arr_dfapp) > 1
            if max(abs(arr_dfapp[end] - arr_dfapp[end-1]), abs(arr_dfcov[end] - arr_dfcov[end-1])) < tol
                break
            end
        end
        iter += 1
        println("iter = $iter")
        if iter > maxiter 
            break
        end
    end
    if trace_plot
        fig = plot(arr_dfcov, label = "df_cov", title = "n = $n, p = $p, nk = $nk")
        plot!(fig, arr_dfapp, label = "df_app")
        plot!(fig, arr_penalty, label = "penalty")
        return fig
    end
    return arr_dfcov, arr_dfapp, arr_penalty
end

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

function vary_p(d = 1, ps = [2, 10, 20, 30, 40, 50, 60])
#    ps = [10]
    RES = zeros(length(ps), 7)
    for (i, p) in enumerate(ps)
        RES[i, :] .= compare_mse(p = p, nrep = 10, d = d)
    end
    return RES
end

function mars_experiment_mse(; folder = "/tmp", ps = [2, 10, 20, 30, 40, 50, 60])
    RES1 = vary_p(1, ps)
    serialize(joinpath(folder, "mars_varyp_d1.sil"), RES1)
    RES2 = vary_p(2, ps)
    serialize(joinpath(folder, "mars_varyp_d1.sil"), RES2)

    N = size(RES1, 1)
    p1 = plot(ps, RES1[:, 3], yerror = RES1[:, 5] ./ sqrt(N), xlab = L"p", ylab = L"R^2", 
                xticks = (ps, string.(ps)), label = "default", title = "degree = 1",
                # legend = false # shared with p2
    )
    plot!(p1, ps, RES1[:, 4], yerror = RES1[:, 6] ./ sqrt(N), label = "corrected")
    savefig("$folder/mars_p1.pdf")
    p2 = plot(ps, RES2[:, 3], yerror = RES2[:, 5] ./ sqrt(N), xlab = L"p", ylab = L"R^2", xticks = (ps, string.(ps)), label = "default", title = "degree = 2", 
        # ylim = [0, 1.0]
        legend = false # share with p1
        )
    plot!(p2, ps, RES2[:, 4], yerror = RES2[:, 6] ./ sqrt(N), label = "corrected")
    savefig("$folder/mars_p2.pdf")
    cols = [:blue :orange]
    p3 = plot(ps, RES1[:, 7], xticks = (ps, string.(ps)), xlab = L"p", ylab = "penalty", label = "corrected (degree = 1)", color = cols[1], markershape = :star5, legend=:topleft)
    plot!(p3, ps, RES2[:, 7], label = "corrected (degree = 2)", color = cols[2], markershape = :dtriangle)
    hline!(p3, [2 3], label = ["default (degree = 1)" "default (degree = 2)"], style = :dash, color = cols)
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
    plot(p1, p11, p2, p21, p3, layout=@layout([grid(2, 1){0.3w} _ grid(2, 1){0.3w} b{0.33w}]), size = (1300, 400)) # too small margin overlap legend
    savefig("$folder/mars_sim.pdf")
end

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