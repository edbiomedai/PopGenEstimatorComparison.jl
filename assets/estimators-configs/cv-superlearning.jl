resampling = JointStratifiedCV(patterns=[r"^rs[0-9]+"], resampling=StratifiedCV(nfolds=3))

xgboost_classifiers = (;(Symbol("xgboost_classifier_", id) => XGBoostClassifier(tree_method="hist", max_depth=max_depth, eta=η, num_round=100) 
  for (id, (max_depth, η)) ∈ enumerate(Iterators.product([2, 4, 6, 8], [0.001, 0.01, 0.3])))...)

xgboost_regressors = (;(Symbol("xgboost_regressor_", id) => XGBoostRegressor(tree_method="hist", max_depth=max_depth, eta=η, num_round=100) 
  for (id, (max_depth, η)) ∈ enumerate(Iterators.product([2, 4, 6, 8], [0.001, 0.01, 0.3])))...)

    
default_models = TMLE.default_models(
  # For the estimation of E[Y|W, T]: continuous outcome
  Q_continuous = Stack(;
    metalearner        = LinearRegressor(fit_intercept=false),
    resampling         = resampling,
    cache              = false,
    glmnet             = Pipeline(
      RestrictedInteractionTransformer(order=2, primary_variables_patterns=[r"^rs[0-9]+"]),
      GLMNetRegressor(resampling=resampling),
      cache = false
    ),
    lr                 = Pipeline(
      RestrictedInteractionTransformer(order=2, primary_variables_patterns=[r"^rs[0-9]+"]),
      LinearRegressor(),
      cache = false
    ),
    xgboost_regressors...
    ),
  # For the estimation of E[Y|W, T]: binary target
  Q_binary = Stack(;
    metalearner        = LogisticClassifier(lambda=0., fit_intercept=false),
    resampling         = resampling,
    cache              = false,
    glmnet             = Pipeline(
      RestrictedInteractionTransformer(order=2, primary_variables_patterns=[r"^rs[0-9]+"]),
      GLMNetClassifier(resampling=resampling),
      cache = false
    ),
    lr                 = Pipeline(
      RestrictedInteractionTransformer(order=2, primary_variables_patterns=[r"^rs[0-9]+"]),
      LogisticClassifier(lambda=0.),
      cache = false
    ),
    xgboost_classifiers...
  ),
  # For the estimation of p(T| W)
  G = Stack(;
    metalearner        = LogisticClassifier(lambda=0., fit_intercept=false),
    resampling         = resampling,
    cache              = false,
    glmnet             = GLMNetClassifier(resampling=resampling),
    lr                 = LogisticClassifier(lambda=0.),
    xgboost_classifiers...
  )
)

ESTIMATORS = (
  CV_wTMLE_SL = TMLEE(models=default_models, weighted=true, resampling=resampling),
  CV_TMLE_SL = TMLEE(models=default_models, weighted=false, resampling=resampling),
  CV_OSE_SL = OSE(models=default_models, resampling=resampling)
)