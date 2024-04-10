xgboost_classifier = XGBoostClassifier(tree_method="hist")
xgboost_regressor = XGBoostRegressor(tree_method="hist")
resampling = JointStratifiedCV(patterns=[r"^rs[0-9]+"], resampling=StratifiedCV(nfolds=3))

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
  G = Stack(
    metalearner        = LogisticClassifier(lambda=0., fit_intercept=false),
    resampling         = resampling,
    cache              = false,
    glmnet             = GLMNetClassifier(resampling=resampling),
    lr                 = LogisticClassifier(lambda=0.),
    tuned_xgboost      = TunedModel(
        model = xgboost_classifier,
        resampling = resampling,
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
  wTMLE_G_SL_Q_GLMNET = TMLEE(models=default_models, weighted=true),
  TMLE_G_SL_Q_GLMNET = TMLEE(models=default_models, weighted=false),
  OSE_G_SL_Q_GLMNET = OSE(models=default_models)
)