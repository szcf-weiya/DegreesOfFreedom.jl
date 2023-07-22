using Statistics
using Printf

function calc_rss(yleft::AbstractVector, yright::AbstractVector)
    return sum((yleft .- mean(yleft)).^2) + sum((yright .- mean(yright)).^2)
end

function find_best_rule(x::AbstractMatrix, y::AbstractVector)
    bst_j, bst_s, bst_rss = nothing, nothing, Inf
    for j in axes(x, 2)
        thresh = sort(unique(x[:, j]))[2:end]
        for s in thresh
            left_idx = x[:, j] .< s
            rss = calc_rss(y[left_idx], y[.!left_idx])
            if rss < bst_rss
                bst_rss = rss
                bst_j = j
                bst_s = s
                # println("j = $j, s = $s, bst_rss = $rss")
            end
        end
    end
    return Dict{Any,Any}("j"=>bst_j,"s"=>bst_s)
end

function split(x::AbstractMatrix, y::AbstractVector, depth::Int, max_depth::Int)
    if depth == max_depth || size(x, 1) < 2
        return mean(y)
    end
    rule = find_best_rule(x, y)
    left_idx = x[:, rule["j"]] .< rule["s"]
    rule["left"] = split(x[left_idx, :], y[left_idx], depth+1, max_depth)
    rule["right"] = split(x[.!left_idx, :], y[.!left_idx], depth+1, max_depth)
    return rule
end

function predict(x::AbstractMatrix, RULES::Union{Dict, Float64})
    pred = zeros(size(x, 1))
    for i in axes(x, 1)
        rules = deepcopy(RULES)
        while typeof(rules) != Float64
            j, s = rules["j"], rules["s"]
            if x[i, j] < s
                rules = rules["left"]
            else
                rules = rules["right"]
            end
        end
        pred[i] = rules
    end
    return pred
end

function calc_df(;n = 100, p = 10, N = 100, maxdepth = 3)
    #n = 100; p = 10
    x = randn(n, p)
    y = randn(n, N)
    yhat = zeros(n, N)
    for i = 1:N
        rule = split(x, y[:, i], 0, maxdepth)
        yhat[:, i] = predict(x, rule)
    end
    return sum([cov(y[i, :], yhat[i, :]) for i=1:n])
    #return [cov(y[i, :], yhat[i, :]) for i=1:n]
end

# only for maxdepth=1 and p = 1
function count_s(;n = 100, p = 1, N = 100, maxdepth = 1)
    x = randn(n, p)
    y = randn(n, N)
    ss = zeros(N)
    for i = 1:N
        rule = split(x, y[:, i], 0, maxdepth)
        ss[i] = rule["s"]
    end
    return x, ss
end

function rep_calc_df(nrep = 10; kw...)
    dfs = zeros(nrep)
    for i = 1:nrep
        dfs[i] = calc_df(;kw...)
    end
    return mean(dfs), std(dfs)/sqrt(nrep)
end

"""
    df_regtree(; ps = [1, 5, 10], maxd = 4)

Experiment for degrees of freedom for regression trees with number of features `ps` and maximum depth `maxd`.
"""
function df_regtree(; ps = [1, 5, 10], maxd = 4, nrep = 10)
    np = length(ps)
    nd = maxd + 1
    res = zeros(nd * np, 2)
    for (j, p) in enumerate(ps)
        for i in 0:maxd
            res[nd * (j-1)+i+1, :] .= rep_calc_df(nrep, p=p, maxdepth=i)
        end
    end
    return res
end

function vary_depth(; p = 10)
    res = zeros(4)
    for d in 1:4
        println("d = $d")
        res[d] = rep_calc_df(3, n = 100, p = p, N = 1000, maxdepth = d)
    end
    return res
end

"""
    run_experiment_tree()

Run the experiment for regression tree, whose results are saved into a `.sil` file (can be later loaded via `deserialize`) and a `.tex` file (the table in the paper).
"""
function run_experiment_tree(; folder = "/tmp", # "../res/df"
                             kw...)
    res = df_regtree(; kw...)
    timestamp = replace(strip(read(`date -Iseconds`, String)), ":" => "_")
    serialize(joinpath(folder, "regtrees_$timestamp.sil"), res)
    print2tex_tree(res, folder)    
end

function print2tex_tree(res0, folder="../res/df", filename = "regtrees.tex")
    res = round.(res0, sigdigits = 4)
    file = joinpath(folder, filename)
    @info "Write table results into $file"
    # open(file, "w") do io
    #     write(io, raw"\begin{tabular}{lccc}", "\n")
    #     writeline(io, raw"\toprule")
    #     writeline(io, raw"$p$ & depth & $M$ & $\hat\df$\tabularnewline")
    #     for (j, p) in enumerate([1, 5, 10])
    #         writeline(io, raw"\midrule")
    #         write(io, raw"\multirow{5}{*}{" * "$p}")
    #         for i in 0:4
    #             writeline(io, "& $i & $(2^i) & $(res[5(j-1)+i+1,1]) ($(res[5(j-1)+i+1,2]))\\tabularnewline")
    #         end
    #     end
    #     writeline(io, raw"\bottomrule")
    #     writeline(io, raw"\end{tabular}")
    # end
    open(file, "w") do io
        write(io, raw"\begin{tabular}{lrrrr}", "\n")
        writeline(io, raw"\toprule")
        writeline(io, raw"\multirow{2}{*}{depth} & \multirow{2}{*}{$M$} & \multicolumn{3}{c}{$\hat\df$}\tabularnewline")
        writeline(io, raw"\cmidrule{3-5}")
        writeline(io, raw"& & $p=1$ & $p=5$ & $p=10$\tabularnewline")
        writeline(io, raw"\midrule")
        for i in 0:4
            write(io, "$i & $(2^i)")
            for j in 1:3
                write(io, "& $(@sprintf "%.2f" res[5(j-1)+i+1,1]) ($(@sprintf "%.2f" res[5(j-1)+i+1,2]))")
            end
            writeline(io, raw"\tabularnewline")
        end
        writeline(io, raw"\bottomrule")
        writeline(io, raw"\end{tabular}")
    end
end
