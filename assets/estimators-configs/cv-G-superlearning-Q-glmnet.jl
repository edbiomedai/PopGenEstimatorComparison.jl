resampling = JointStratifiedCV(patterns=[r"^rs[0-9]+"], resampling=StratifiedCV(nfolds=3))

xgboost_classifiers = (;(Symbol("xgboost_classifier_", id) => XGBoostClassifier(tree_method="hist", max_depth=max_depth, eta=η, num_round=100) 
  for (id, (max_depth, η)) ∈ enumerate(Iterators.product([2, 4, 6, 8], [0.001, 0.01, 0.3])))...)

default_models = TMLE.default_models(
  # For the estimation of E[Y|W, T]: continuous outcome
  Q_continuous = Pipeline(
    RestrictedInteractionTransformer(order=2, primary_variables_patterns=[r"^rs[0-9]+"]),
    GLMNetRegressor(resampling=resampling),
    cache = false
  ),
  # For the estimation of E[Y|W, T]: binary outcome
  Q_binary = Pipeline(
    RestrictedInteractionTransformer(order=2, primary_variables_patterns=[r"^rs[0-9]+"]),
    GLMNetClassifier(resampling=resampling),
    cache = false
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
  CV_wTMLE_G_SL_Q_GLMNET = TMLEE(models=default_models, weighted=true, resampling=resampling),
  CV_TMLE_G_SL_Q_GLMNET = TMLEE(models=default_models, weighted=false, resampling=resampling),
  CV_OSE_G_SL_Q_GLMNET = OSE(models=default_models, resampling=resampling)
)