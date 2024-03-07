
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

function linear_interaction_dataset(n=100)
    G₀ = Normal()
    U  = Uniform(0, 1)
    W = rand(G₀, n)
    C = rand(G₀, n)
    μT₁ = logistic.(2W .+ 1)
    T₁ = Int.(rand(U, n) .< μT₁)
    μT₂ = logistic.(-1W .+ 0.5)
    T₂ = Int.(rand(U, n) .< μT₂)
    μYcont = 0.5T₁ .- 1.2T₂ .+ 0.3T₁.*T₂ .+ 0.6W .- 0.2C
    Ycont = μYcont .+ rand(G₀, n) 
    μYbin = logistic.(0.5T₁ .- 1.2T₂ .+ 0.3T₁.*T₂ .+ 0.6W .- 0.2C)
    Ybin = Int.(rand(U) .< μYbin)
    return DataFrame(
        W =W, 
        C =C, 
        T₁ = T₁, 
        T₂ = T₂, 
        Ycont = Ycont, 
        Ybin = Ybin
    )
end

function linear_interaction_dataset_ATEs()
    return TMLE.Configuration(
        estimands = [
        ATE(
            outcome=:Ycont,
            treatment_values = (T₁ = (case=1, control=0),),
            treatment_confounders = (:W,),
            outcome_extra_covariates = (:C,)
        ),
        ATE(
            outcome=:Ybin,
            treatment_values = (T₁ = (case=1, control=0), T₂ = (case=1, control=0)),
            treatment_confounders = (:W,),
            outcome_extra_covariates = (:C,)
        )
    ])
end

function linear_interaction_dataset_IATEs()
    return TMLE.Configuration(
        estimands = [
        IATE(
            outcome=:Ycont,
            treatment_values = (T₁ = (case=1, control=0), T₂ = (case=1, control=0)),
            treatment_confounders = (:W,),
            outcome_extra_covariates = (:C,)
        ),
        IATE(
            outcome=:Ybin,
            treatment_values = (T₁ = (case=1, control=0), T₂ = (case=1, control=0)),
            treatment_confounders = (:W,),
            outcome_extra_covariates = (:C,)
        )
    ])
end

function write_linear_interaction_dataset_estimands()
    serialize("test/assets/estimands/iates.jls", linear_interaction_dataset_IATEs())
    serialize("test/assets/estimands/ates.jls", linear_interaction_dataset_ATEs())
end

