using RCall
using Statistics
using Plots
using LinearAlgebra
import MonotoneSplines # only module name, avoid conflict with its export predict
using Serialization
using LaTeXStrings
using Printf

function linear_indep_rows(X::AbstractMatrix, tol = 1e-10)
    n, p = size(X)
    d = qr(X', ColumnNorm()) # https://en.wikipedia.org/wiki/QR_decomposition#Column_pivoting
    # check the leading non-zeros
    idx = abs.(diag(d.R)) .> tol 
    raw_idx = d.p[findall(idx)] # note that length of p might be larger than length of idx, but the nonzero index should be ok
    ## the following still problematic, cannot avoid the case when all zeros in one row, but not zero in the later positions
    # ir = 1 # index for row of R, starting from the first row
    # for i = 1:n # index for column of R
    #     println("checking ($ir, $i)")
    #     if abs(d.R[ir, i]) > tol
    #         append!(idx, i)
    #         ir += 1
    #     end
    #     if ir > p
    #         break
    #     end
    # end
    return raw_idx
end

function calc_df_spl(; method = "lm", J = 4, λ = 1, verbose = false,
                   n = 20, nMC = 1000, tol = 1e-6, p = 10, calc_theoretical_df = false)
    y = randn(n, nMC)
    x = collect(1.0:n)
    # x = randn(n, p)
    yhat = zeros(n, nMC)
    # theoretical df
    df = nothing
    dfs = zeros(nMC)
    if method == "lm"
        df = 2
        for i = 1:nMC
            yhat[:, i] = rcopy(R"lm($(y[:,i]) ~ $x)$fitted.values")
        end
    elseif method == "cubic_spline"
        # cubic spline
        df = J
        for i = 1:nMC
            yhat[:, i] = MonotoneSplines.predict(MonotoneSplines.bspline(x, y[:, i], J), x)
        end
    elseif method == "cubic_smooth_spline"
        for i = 1:nMC
            # spl = R"smooth.spline($x, $(y[:,i]), lambda = $lambda)"
            # number of basis functions J = K + M = K + 4
            # where K is the number of internal knots, K = nknots - 2,
            # so J = nknots + 2
            # spl = R"smooth.spline($x, $(y[:,i]), lambda = $lambda, nknots = $(J-2))"
            # fix nknots = n
            lam = λ # see #2
            spl = R"smooth.spline($x, $(y[:,i]), lambda = $lam, nknots = $(n))"
            if i == 1
                df = rcopy(R"$spl$df")
            end
            yhat[:, i] = rcopy(R"predict($spl)$y")
        end
    elseif method == "monotone_cubic_spline"
        for i = 1:nMC
            monocs = MonotoneSplines.mono_cs(x, y[:, i], J)
            yhat[:, i] = monocs.fitted
            neq = sum(abs.(monocs.β[1:end-1] - monocs.β[2:end]) .< tol)
            dfs[i] = J - neq
        end
        df = mean(dfs)
    elseif method == "monotone_smooth_spline"
        for i = 1:nMC
            monoss = MonotoneSplines.mono_ss(x, y[:, i], λ)
            yhat[:, i] = monoss.fitted
            if calc_theoretical_df
                idx_eq = abs.(monoss.β[1:end-1] - monoss.β[2:end]) .< tol
                n_eq = sum(idx_eq)
                J = length(monoss.β)
                K = diagm(0 => ones(J), 1 => -ones(J-1))[1:J-1,:]
                # construct A and B matrix
                A = vcat(1.0I(J), -1.0I(J), zeros(n_eq, J))
                B = vcat(-monoss.L', monoss.L', K[idx_eq, :]) * inv(monoss.B' * monoss.B) * monoss.B'
                AB = hcat(A, B)
                # find linearly independent rows
                idx = linear_indep_rows(AB)
                # println("number of indep rows: $(size(AB, 1)) -> $(length(idx))")
                #idx = round.(diag(D.R), digits=6) .!= 0 # not necessarily diagonal. instead the learning element for each row
                A1 = AB[idx, 1:J]
                B1 = AB[idx, (J+1):end]
                dfs[i] = n - tr(B1' * inv(B1 * B1' + A1 * A1' / λ) * B1)
            end
        end
        df = mean(dfs)
        # elseif method == "tree"
    #     ms = zeros(nMC)
    #     for i = 1:nMC
    #         treefit = R"rpart::rpart($(y[:,i]) ~ $x)"
    #         yhat[:, i] = rcopy(R"predict($treefit)")
    #         #yhat[:, i] = rcopy(R"predict(rpart::rpart($(y[:,i]) ~ $x))")
    #         ms[i] = rcopy(R"length(unique($treefit$where))")
    #     end
    #     println(quantile(ms)')
    #     println(mean(ms))
    end
    if verbose
        println("theoretical df = $df")
    end
    sum([cov(y[i, :], yhat[i, :]) for i = 1:n]), df, dfs
end

function rep_calc_df_spl(nrep = 10; fig = false, kw...)
    dfs = zeros(nrep)
    df = nothing
    for j = 1:nrep
        if j == 1
            dfs[j], df = calc_df_spl(; kw...)
        else
            dfs[j] = calc_df_spl(; kw...)[1]
        end
    end
    μ, σ = mean(dfs), std(dfs)/sqrt(nrep)
    if fig
        histogram(dfs, title = "μ = $(round(μ, digits = 3)), σ = $(round(σ, digits = 3))", legend = false)
    else
        μ, σ, df
    end
end

# estimate df of cubic monotone spline
function est_df(method = "monotone_cubic_spline"; Js = 4:30, nMC = 100, nrep = 100, n = 20, kw...)
    res = zeros(length(Js), 3)
    for (i, J) in enumerate(Js)
        println("J = $J")
        try
            res[i, :] .= rep_calc_df_spl(nrep; method = method, J = J, fig=false, nMC = nMC, n = n, kw...)
        catch e
            println(e)
        end
    end
    return res
end

# function plotdf()
#     df_ms = deserialize("tmpres/df-4to100-100-100-100.sil")
#     Js = 4:100
#     plot(Js, df_ms[:, 1], yerror = df_ms[:, 2], label = "monotone", legend = :topleft)
#     plot!(Js, log.(Js) .+ 1/2, label = "log(J) + 1/2", lw=3)
# end
"""
    df_splines(; Js = [5, 10, 15], λs = [0.001, 0.01, 0.1], n = 20, nrep = 10, nMC = 100)

Calculate the empirical degrees of freedom of four splines:
- cubic splines with number of basis functions `Js`
- smoothing splines with tuning parameter `λs`
- sample size `n`
- number of repetition `nrep`
- number of Monte Carlo samples `nMC`
"""
function df_splines(; Js = [5, 10, 15], λs = [0.001, 0.01, 0.1], n = 20, nrep = 10, nMC = 100)
    nJ = length(Js)
    nλ = length(λs)
    res = zeros(nJ*2 + nλ*2, 3)
    # cubic spline
    for (j, J) in enumerate(Js)
        res[j, :] .= rep_calc_df_spl(nrep, method="cubic_spline", J = J, n = n, nMC = nMC)
        res[nJ + j, :] .= rep_calc_df_spl(nrep, method="monotone_cubic_spline", J = J, n = n, nMC = nMC)
    end
    for (i, λ) in enumerate(λs)
        res[2nJ + i, :] .= rep_calc_df_spl(nrep, method="cubic_smooth_spline", λ = λ, n = n, nMC = nMC)
        res[2nJ + nλ + i, :] .= rep_calc_df_spl(nrep, method="monotone_smooth_spline", λ = λ, n = n, nMC = nMC)
    end
    return res
end

"""
    run_experiment_splines()

Run the experiment of splines, whose results will be saved into a `.sil` file (can be later loaded via `deserialize`) and a `.tex` file (the table displayed in the paper).

```julia
run_experiment_splines(folder = "/home/weiya/Overleaf/paperDoF/res/df")
```
"""
function run_experiment_splines(; folder = "/tmp", # "../res/df"
                                nMC = 100, n = 100, nrep = 100)
    res = df_splines(nrep = nrep, nMC = nMC, n = n)
    filename = "splines-nrep$nrep-nMC$nMC-n$n"
    serialize(joinpath(folder, "$filename.sil"), res)
    # if needed, load the save results as follows
    #res = deserialize("../res/df/splines2023-01-24T22_38_22-05_00.sil")
    print2tex_spl(res, [5, 10, 15], [0.001, 0.01, 0.1], folder, "$filename.tex")
end

function print2tex_spl(res0, Js, λs, folder = "../res/df", filename = "splines.tex")
    res = round.(res0, sigdigits=4)
    n = size(res, 1)
    nJ = length(Js)
    nλ = length(λs)
    file = joinpath(folder, filename)
    @info "Write table results into $file"
    open(file, "w") do io
        write(io, raw"\begin{tabular}{lrrr}", "\n")
        writeline(io, raw"\toprule")
        writeline(io, raw"Method & Parameter & Theoretical & Empirical\tabularnewline")
        writeline(io, raw"\midrule")
        # cubic spline
        # since raw"\\" stll return `\`, use `tabularnewline` instead
        write(io, raw"\multirow{", "$nJ", raw"}{*}{$\hat\mu^\cubic(J)$}")
        for j = 1:nJ
            writeline(io, "& $(Js[j]) & $(res[j, 3]) & $(@sprintf "%.2f" res[j, 1]) ($(@sprintf "%.3f" res[j, 2]))\\tabularnewline")
        end
        writeline(io, raw"\midrule")
        # smooth spline
        write(io, raw"\multirow{", "$nλ", raw"}{*}{$\hat\mu^\smooth(\lambda)$}")
        for i = 1:nλ
            writeline(io, "& $(λs[i]) & $(@sprintf "%.2f" res[2nJ+i, 3]) & $(@sprintf "%.2f" res[2nJ+i, 1]) ($(@sprintf "%.3f" res[2nJ+i, 2])) \\tabularnewline")
        end
        writeline(io, raw"\midrule")
        # monotone spline
        write(io, raw"\multirow{", "$nJ", raw"}{*}{$\hat\mu^{\mono,\cubic}(J)$}")
        for j = 1:nJ
            writeline(io, "& $(Js[j]) & $(@sprintf "%.2f" res[nJ+j, 3]) & $(@sprintf "%.2f" res[nJ+j, 1]) ($(@sprintf "%.3f" res[nJ+j, 2]))\\tabularnewline")
        end
        writeline(io, raw"\midrule")
        # monotone smoothing spline
        write(io, raw"\multirow{", "$nJ", raw"}{*}{$\hat\mu^{\mono,\smooth}(\lambda)$}")
        for i = 1:nλ
            #writeline(io, "& $(λs[i]) & $(res[2nJ+nλ+i, 3]) & $(res[2nJ+nλ+i, 1]) ($(res[2nJ+nλ+i, 2]))\\tabularnewline")
            writeline(io, "& $(λs[i]) & - & $(@sprintf "%.2f" res[2nJ+nλ+i, 1]) ($(@sprintf "%.3f" res[2nJ+nλ+i, 2]))\\tabularnewline")
        end        
        writeline(io, raw"\bottomrule")
        writeline(io, raw"\end{tabular}")
    end
end
