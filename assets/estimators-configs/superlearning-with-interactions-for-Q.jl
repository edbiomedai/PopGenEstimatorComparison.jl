xgboost_regressor = XGBoostRegressor(tree_method="hist")
xgboost_classifier = XGBoostClassifier(tree_method="hist")

default_models = TMLE.default_models(
  # For the estimation of E[Y|W, T]: continuous outcome
  Q_continuous = Stack(
    metalearner        = LinearRegressor(fit_intercept=false),
    resampling         = CV(nfolds=3),
    cache              = false,
    glmnet             = Pipeline(
      RestrictedInteractionTransformer(order=2, primary_variables_patterns=[r"^rs[0-9]+"]),
      GLMNetRegressor(resampling=CV(nfolds=3)),
      cache = false
    ),
    lr                 = Pipeline(
      RestrictedInteractionTransformer(order=2, primary_variables_patterns=[r"^rs[0-9]+"]),
      LinearRegressor(),
      cache = false
    ),
    tuned_xgboost      = TunedModel(
        model = xgboost_regressor,
        resampling = CV(nfolds=3),
        tuning = Grid(goal=20),
        range = [
            range(xgboost_regressor, :max_depth, lower=3, upper=7), 
            range(xgboost_regressor, :lambda, lower=1e-5, upper=10, scale=:log)
            ],
        measure = rmse,
        cache=false
        )
    ),
  # For the estimation of E[Y|W, T]: binary target
  Q_binary = Stack(
    metalearner        = LogisticClassifier(lambda=0., fit_intercept=false),
    resampling         = StratifiedCV(nfolds=3),
    cache              = false,
    glmnet             = Pipeline(
      RestrictedInteractionTransformer(order=2, primary_variables_patterns=[r"^rs[0-9]+"]),
      GLMNetClassifier(resampling=StratifiedCV(nfolds=3)),
      cache = false
    ),
    lr                 = Pipeline(
      RestrictedInteractionTransformer(order=2, primary_variables_patterns=[r"^rs[0-9]+"]),
      LogisticClassifier(lambda=0.),
      cache = false
    ),
    tuned_xgboost      = TunedModel(
        model = xgboost_classifier,
        resampling = StratifiedCV(nfolds=3),
        tuning = Grid(goal=20),
        range = [
            range(xgboost_classifier, :max_depth, lower=3, upper=7), 
            range(xgboost_classifier, :lambda, lower=1e-5, upper=10, scale=:log)
            ],
        measure = log_loss,
        cache=false
        )
  ),
  # For the estimation of p(T| W)
  G = Stack(
    metalearner        = LogisticClassifier(lambda=0., fit_intercept=false),
    resampling         = StratifiedCV(nfolds=3),
    cache              = false,
    glmnet             = GLMNetClassifier(),
    lr                 = LogisticClassifier(lambda=0.),
    tuned_xgboost      = TunedModel(
        model = xgboost_classifier,
        resampling = StratifiedCV(nfolds=3),
        tuning = Grid(goal=20),
        range = [
            range(xgboost_classifier, :max_depth, lower=3, upper=7), 
            range(xgboost_classifier, :lambda, lower=1e-5, upper=10, scale=:log)
            ],
        measure = log_loss,
        cache=false
    )
  )
)

ESTIMATORS = (
  TMLE = TMLEE(models=default_models, weighted=true, ps_lowerbound=1e-8),
)