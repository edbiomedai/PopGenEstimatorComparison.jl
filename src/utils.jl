
function variables_from_estimand(Ψ)
    return unique(Iterators.flatten(
        [[Ψ.outcome], 
        keys(Ψ.treatment_values), 
        values(Ψ.treatment_confounders)...,
        Ψ.outcome_extra_covariates]
    ))
end