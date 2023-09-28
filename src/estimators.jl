mutable struct LinearEstimator

end

function (estimator::LinearEstimator)(Ψ, dataset; verbosity=1)
    features = [keys(Ψ.treatment_values)..., first(Ψ.treatment_confounders)..., Ψ.outcome_extra_covariates...]
    outcome = Ψ.outcome
    if Ψ isa TMLE.StatisticalATE
        if length(Ψ.treatment_values) > 1
            # Conditional ATE
            eQTL_values = values(Ψ.treatment_values)[2]
            if eQTL_values.case == eQTL_values.control
                eqtl = keys(Ψ.treatment_values)[2]
                train_dataset = filter(x -> x[eqtl] == eQTL_values.case, dataset[!, vcat(outcome, features)])
                features = filter(x -> x !== eqtl, vcat(outcome, features))
            else
                throw(ArgumentError(string("Don't know how to perform linear inference with: ", Ψ)))
            end
        else
            train_dataset = dataset
        end
    end

    X = train_dataset[!, features]
    y = float(train_dataset[!, Ψ.outcome])
    model = OneHotEncoder(drop_last=true) |> MLJGLM.LinearRegressor()
    mach = machine(model, X, y)
    fit!(mach, verbosity=verbosity)
    coef_table = report(mach).linear_regressor.coef_table
end