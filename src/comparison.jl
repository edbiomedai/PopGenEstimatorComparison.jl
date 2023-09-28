# Constants

const nfolds = 3
const cv = CV(nfolds=nfolds)
const stratified_cv = StratifiedCV(nfolds=nfolds)
const xgboost_classifier = XGBoostClassifier(tree_method="hist")
const xgboost_regressor = XGBoostRegressor(tree_method="hist")
const all_models = Dict(
    :GLM_CATEGORICAL => LogisticClassifier(),
    :GLM_CONTINUOUS => LinearRegressor(),
    :GLMNet_CATEGORICAL => GLMNetClassifier(resampling=stratified_cv),
    :GLMNet_CONTINUOUS => GLMNetRegressor(resampling=cv),
    :SL_CATEGORICAL => Stack(
        cache              = false,
        metalearner=LogisticClassifier(fit_intercept=false),
        resampling=stratified_cv,
        glm = LogisticClassifier(),
        glmnet = GLMNetClassifier(resampling=stratified_cv),
        tuned_xgboost      = TunedModel(
        model = xgboost_classifier,
        resampling = stratified_cv,
        tuning = Grid(goal=20),
        range = [
            range(xgboost_classifier, :max_depth, lower=3, upper=7), 
            range(xgboost_classifier, :lambda, lower=1e-5, upper=10, scale=:log)
            ],
        measure = log_loss,
        cache=false
        )
    ),
    :SL_CONTINUOUS => Stack(
        metalearner        = LinearRegressor(fit_intercept=false),
        resampling         = cv,
        cache              = false,
        glmnet             = GLMNetRegressor(resampling=cv),
        lr                 = LinearRegressor(),
        tuned_xgboost      = TunedModel(
            model = xgboost_regressor,
            resampling = cv,
            tuning = Grid(goal=20),
            range = [
                range(xgboost_regressor, :max_depth, lower=3, upper=7), 
                range(xgboost_regressor, :lambda, lower=1e-5, upper=10, scale=:log)
                ],
            measure = rmse,
            cache=false
            )
        ),
)

# Functions

function plot_confints(Ψ, dataset_results)
    fig = Figure(resolution = (1200, 1000))
    treatment_spec = join(
        (string(treatment, ": ", treatment_values.control, "⟶", treatment_values.case) 
        for (treatment, treatment_values) ∈ zip(keys(Ψ.treatment_values), Ψ.treatment_values)), 
        ", "
    )
    title = string(replace(string(typeof(Ψ)), "TMLE.Statistical" => ""), ": ", Ψ.outcome, ", " , treatment_spec)
    for (axis_id, (datasetname, results)) in enumerate(dataset_results)
        estimates = [TMLE.estimate(result[2]) for result in results]
        upper_bounds = [confint(TMLE.OneSampleTTest(result[2]))[2] for result in results]
        nestimators = length(results)
        ax = Axis(fig[1, axis_id], title=datasetname, yticks = (1:nestimators, [x[1] for x ∈ results]))
        errorbars!(ax, 
            estimates,
            1:nestimators, 
            upper_bounds .- estimates,
            whiskerwidth = 15,
            color=1:nestimators,
            colormap=:seaborn_colorblind,
            colorrange = (1, nestimators),
            direction = :x
        )
        scatter!(ax, estimates, 1:nestimators, 
            markersize=15, 
            color=1:nestimators,
            colormap=:seaborn_colorblind,
            colorrange = (1, nestimators)
        )
    end
    Label(fig[0, :], title)
    return fig
end

function estimand_from_results_row(row)
    estimand_type = eval(Symbol(row.PARAMETER_TYPE))
    confounders = split(row.CONFOUNDERS, "_&_")
    outcome_extra_covariates = split(row.COVARIATES, "_&_")
    controls = split(row.CONTROL, "_&_")
    cases = split(row.CASE, "_&_")
    treatments = Tuple(Symbol(x) for x in split(row.TREATMENTS, "_&_"))
    treatment_values = NamedTuple{treatments}([(control=control, case=case) for (control, case) ∈ zip(controls, cases)])
    treatment_confounders = NamedTuple{treatments}([confounders for _ in eachindex(treatments)])
    return estimand_type(
        outcome = row.TARGET,
        treatment_values=treatment_values,
        treatment_confounders = treatment_confounders,
        outcome_extra_covariates = outcome_extra_covariates
    )
end

outcome_is_binary(outcome, dataset) = 
    length(unique(skipmissing(dataset[!, outcome]))) == 2

function outcome_model(models, outcome, dataset; flavor=:GLM)
    target_type = outcome_is_binary(outcome, dataset) ? :CATEGORICAL : :CONTINUOUS
    return with_encoder(models[Symbol(flavor, :_, target_type)])
end

function treatment_model(models; flavor=:GLM)
    return models[Symbol(flavor, :_CATEGORICAL)]
end

function coercetypes!(dataset, Ψ)
    if outcome_is_binary(Ψ.outcome, dataset) 
        dataset[!, Ψ.outcome] = categorical(dataset[!, Ψ.outcome])
    end
    for treatment in keys(Ψ.treatment_values)
        dataset[!, treatment] = categorical(dataset[!, treatment])
    end
    for covariate in (first(Ψ.treatment_confounders)..., Ψ.outcome_extra_covariates...)
        dataset[!, covariate] = float(dataset[!, covariate])
    end
end

function compare_estimators(parsed_args)
    include(abspath(parsed_args["config-file"]))
    verbosity = parsed_args["verbosity"]
    # Load Data
    problematic_estimands = CSV.read(estimands_file, DataFrame)
    # Estimand
    row = problematic_estimands[parsed_args["id"], :]
    Ψ = estimand_from_results_row(row)
    dataset_results = Dict()
    for (datasetname, datasetfile) in datasets
        dataset = DataFrame(Arrow.Table(datasetfile))
        # Coerce datatypes
        coercetypes!(dataset, Ψ)
        # Estimators
        model_keys = tuple(Ψ.outcome, keys(Ψ.treatment_values)...)
        glm_η̂s = NamedTuple{model_keys}(
            [outcome_model(all_models, Ψ.outcome, dataset; flavor=:GLM), (treatment_model(all_models, flavor=:GLM) for _ in 1:length(model_keys) - 1)...]
        )
        glmnet_η̂s = NamedTuple{model_keys}(
            [outcome_model(all_models, Ψ.outcome, dataset; flavor=:GLMNet), (treatment_model(all_models, flavor=:GLMNet) for _ in 1:length(model_keys) - 1)...]
        )
        sl_η̂s = NamedTuple{model_keys}(
            [outcome_model(all_models, Ψ.outcome, dataset; flavor=:SL), (treatment_model(all_models, flavor=:SL) for _ in 1:length(model_keys) - 1)...]
        )
        Ψ̂s = (
            # SL models
            "TMLE(SL)" => TMLEE(sl_η̂s, weighted=false,),
            "Weighted-TMLE(SL)" => TMLEE(sl_η̂s, weighted=true),
            "OSE(SL)" => OSE(sl_η̂s),
            "CV-TMLE(SL)" => TMLEE(sl_η̂s, weighted=false, resampling=cv),
            "Weighted-CV-TMLE(SL)" => TMLEE(sl_η̂s, weighted=true, resampling=cv),
            "CV-OSE(SL)" => OSE(sl_η̂s, resampling=cv),
            # GLMNet models
            "TMLE(GLMNet)" => TMLEE(glmnet_η̂s, weighted=false,),
            "Weighted-TMLE(GLMNet)" => TMLEE(glmnet_η̂s, weighted=true),
            "OSE(GLMNet)" => OSE(glmnet_η̂s),
            "CV-TMLE(GLMNet)" => TMLEE(glmnet_η̂s, weighted=false, resampling=cv),
            "Weighted-CV-TMLE(GLMNet)" => TMLEE(glmnet_η̂s, weighted=true, resampling=cv),
            "CV-OSE(GLMNet)" => OSE(glmnet_η̂s, resampling=cv),
            # GLM models
            "TMLE(GLM)" => TMLEE(glm_η̂s, weighted=false,),
            "Weighted-TMLE(GLM)" => TMLEE(glm_η̂s, weighted=true),
            "OSE(GLM)" => OSE(glm_η̂s),
            "CV-TMLE(GLM)" => TMLEE(glm_η̂s, weighted=false, resampling=cv),
            "Weighted-CV-TMLE(GLM)" => TMLEE(glm_η̂s, weighted=true, resampling=cv),
            "CV-OSE(GLM)" => OSE(glm_η̂s, resampling=cv),
        )
        # Estimation
        cache = Dict()
        results = []
        for (Ψ̂name, Ψ̂) ∈ Ψ̂s
            verbosity > 0 && @info string("Estimating Ψ with: ", Ψ̂name, ".")
            result, cache = Ψ̂(Ψ, dataset, cache=cache, verbosity=verbosity-1)
            push!(results, Ψ̂name => result)
        end
        dataset_results[datasetname] = results
    end
    # plot_cis
    fig = plot_confints(Ψ, dataset_results)
    save(parsed_args["out"], fig)
end




