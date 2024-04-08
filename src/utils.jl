########################################################################
###                          Misc Functions                          ###
########################################################################

continuous_encoder() = ContinuousEncoder(drop_last=true)

getlabels(col::CategoricalVector) = levels(col)
getlabels(col) = nothing

get_input_size(::Type{Multiclass{N}}) where N = N - 1 
get_input_size(::Type) = 1
get_input_size(x::AbstractVector) = get_input_size(elscitype(x))
get_input_size(X) = sum(get_input_size(x) for x in eachcol(X))

propensity_score_inputs(variables) = collect(variables.confounders)
outcome_model_inputs(variables) = vcat(collect(variables.treatments), collect(variables.confounders), collect(variables.outcome_extra_covariates))
confounders_and_covariates(variables) = vcat(collect(variables.confounders), collect(variables.outcome_extra_covariates))

function sample_from(dataset::DataFrame, variables; n=100)
    sample_rows = StatsBase.sample(1:nrow(dataset), n, replace=true)
    return dataset[sample_rows, collect(variables)]
end

variables_from_args(outcome, treatments, confounders, outcome_extra_covariates) = (
    outcome = Symbol(outcome),
    treatments = Symbol.(Tuple(treatments)),
    confounders = Symbol.(Tuple(confounders)),
    outcome_extra_covariates = Symbol.(Tuple(outcome_extra_covariates))
    )

transpose_target(y, labels) = onehotbatch(y, labels)
transpose_target(y, ::Nothing) = Float32.(reshape(y, 1, length(y)))

transpose_table(X) = Float32.(Tables.matrix(X, transpose=true))
transpose_table(estimator, X) =
    transpose_table(MLJBase.transform(estimator.encoder, X))


function get_conditional_densities_variables(estimands)
    conditional_densities_variables = Set{Pair}([])
    for Ψ in estimands
        for factor in TMLE.nuisance_functions_iterator(Ψ)
            push!(conditional_densities_variables, factor.parents => factor.outcome)
        end
    end
    return [Dict("outcome" => pair[2], "parents" => collect(pair[1])) for pair in conditional_densities_variables]
end

function compute_statistics(dataset, Ψ::TMLE.Estimand)
    outcome = get_outcome(Ψ)
    treatments = get_treatments(Ψ)
    nomissing_dataset = dropmissing(dataset, [outcome, treatments..., confounders_and_covariates_set(Ψ)...])
    categorical_variables = autotype(dataset[!, outcome]) <: Finite ? (outcome, treatments...) : treatments

    statistics = Dict()
    # Each Variable
    for variable ∈ categorical_variables
        statistics[variable] = DataFrames.combine(groupby(nomissing_dataset, variable), proprow, nrow)
    end
    # Joint treatment
    if length(treatments) > 1
        statistics[treatments] = DataFrames.combine(groupby(nomissing_dataset, collect(treatments)), proprow, nrow)
    end
    # Joint treatment/outcome
    if length(categorical_variables) > length(treatments)
        statistics[categorical_variables] = DataFrames.combine(groupby(nomissing_dataset, collect(categorical_variables)), proprow, nrow)
    end
    return statistics
end

compute_statistics(dataset, estimands) =
    [compute_statistics(dataset, Ψ) for Ψ in estimands]

########################################################################
###                    Train / Validation Splits                     ###
########################################################################

function stratified_holdout_train_val_samples(X, y;
    resampling=JointStratifiedCV(patterns=[r"^rs[0-9]+"], resampling=StratifiedCV(nfolds=10))
    )
    first(MLJBase.train_test_pairs(resampling, 1:length(y), X, y))
end

function train_validation_split(X, y; 
    train_ratio=10, 
    resampling=JointStratifiedCV(patterns=[r"^rs[0-9]+"], resampling=StratifiedCV(nfolds=train_ratio))
    )
    # Get Train/Validation Splits
    train_samples, val_samples = stratified_holdout_train_val_samples(X, y; resampling=resampling)
    # Split Data
    X_train = selectrows(X, train_samples)
    X_val = selectrows(X, val_samples)
    y_train = selectrows(y, train_samples)
    y_val = selectrows(y, val_samples)

    return (X_train, y_train, X_val, y_val)
end

########################################################################
###                    Results Files Manipulation                    ###
########################################################################

function read_results_file(file)
    jldopen(file) do io
        return reduce(vcat, (io[key] for key in keys(io)))
    end
end

repeat_filename(outdir, repeat) = joinpath(outdir, string("output_", repeat, ".hdf5"))

function read_results_dir(outdir)
    results = []
    for filename in readdir(outdir, join=true)
        repeat_id = parse(Int, split(replace(filename, ".hdf5" => ""), "_")[end])
        fileresults = read_results_file(filename)
        fileresults = [merge(result, (REPEAT_ID=repeat_id,)) for result in fileresults]
        append!(results, fileresults)
    end
    
    return DataFrame(results)
end

read_df_result(file) = jldopen(io -> io["results"], file)

read_df_results(outfiles...) = reduce(vcat, read_df_result(f) for f in outfiles)

function save_aggregated_df_results(input_prefix, out)
    dir = dirname(input_prefix)
    dir = dir !== "" ? dir : "."
    baseprefix = basename(input_prefix)
    results = reduce(vcat, read_df_result(joinpath(dir, file)) for file in readdir(dir) if startswith(file, baseprefix))
    jldsave(out, results=results)
end

########################################################################
###                    Estimand variables accessors                  ###
########################################################################

function confounders_and_covariates_set(Ψ)
    confounders_and_covariates = Set{Symbol}([])
    push!(
        confounders_and_covariates, 
        Iterators.flatten(Ψ.treatment_confounders)..., 
        Ψ.outcome_extra_covariates...
    )
    return confounders_and_covariates
end

confounders_and_covariates_set(Ψ::ComposedEstimand) = 
    union((confounders_and_covariates_set(arg) for arg in Ψ.args)...)

get_outcome(Ψ) = Ψ.outcome

function get_outcome(Ψ::ComposedEstimand)
    @assert Ψ.f == TMLE.joint_estimand "Only joint estimands can be processed at the moment."
    outcome = get_outcome(first(Ψ.args))
    @assert all(get_outcome(x) == outcome for x in Ψ.args)
    return outcome
end

get_treatments(Ψ) = keys(Ψ.treatment_values)

function get_treatments(Ψ::ComposedEstimand) 
    @assert Ψ.f == TMLE.joint_estimand "Only joint estimands can be processed at the moment."
    treatments = get_treatments(first(Ψ.args))
    @assert all(get_treatments(x) == treatments for x in Ψ.args)
    return treatments
end