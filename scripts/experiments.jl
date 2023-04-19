using Serialization
using Printf

function run_experiment()
    res = df_regtree()
    timestamp = replace(strip(read(`date -Iseconds`, String)), ":" => "_")
    serialize("../res/df/regtrees_$timestamp.sil", res)
    print2tex(res)
end

writeline(io, str...) = write(io, str..., "\n")

function print2tex(res0, folder="../res/df", filename = "regtrees.tex")
    res = round.(res0, sigdigits = 4)
    file = joinpath(folder, filename)
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
