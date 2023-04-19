using Statistics

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
function df_regtree(; ps = [1, 5, 10], maxd = 4)
    np = length(ps)
    nd = maxd + 1
    res = zeros(nd * np, 2)
    for (j, p) in enumerate(ps)
        for i in 0:maxd
            res[nd * (j-1)+i+1, :] .= rep_calc_df(p=p, maxdepth=i)
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

