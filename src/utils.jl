"""
    save_plots(ps; output)
Save multi-images into a pdf file, if `output` is unspecified (default), the resulting file is `/tmp/all.pdf`.
See also: <https://github.com/szcf-weiya/Xfunc.jl/blob/master/src/plot.jl>
"""
function save_plots(ps::Array; output = nothing, tmp = tempdir())
    n = length(ps)
    for (i, p) in enumerate(ps)
        savefig(p, "$tmp/p$i.pdf")
    end
    fignames = "$tmp/p" .* string.(1:n) .* ".pdf"
    run(`pdftk $fignames cat output $tmp/all.pdf`)
    if !isnothing(output)
        mv("$tmp/all.pdf", output, force = true)
    end
end

# Tip: convert tuple (a, b, c) to array [a, b, c] via `collect`
save_plots(ps::Tuple; kw...) = save_plots(collect(ps); kw...)

# used in print2tex
writeline(io, str...) = write(io, str..., "\n")