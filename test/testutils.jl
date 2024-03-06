
function dummy_dataset(n=100)
    return DataFrame(
        T₁ = categorical(rand(["AC", "CC", "AA"], n)),
        T₂ = categorical(rand(["GT", "GG", "TT"], n)),
        Ybin = categorical(rand([0, 1], n)),
        Ycont = rand(n),
        W₁ = rand(n),
        W₂ = rand(n),
        C = rand(n)
    )
end

function sinusoidal_dataset(;n_samples=100)
    ϵ = rand(Normal(), n_samples)
    x = rand(Uniform(-10.5, 10.5), n_samples)
    y = 7sin.(0.75x) .+ 0.5x .+ ϵ
    return DataFrame(x=x, y=y)
end