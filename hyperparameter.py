from sklearn.model_selection import RandomizedSearchCV


def tune_model(pipeline, model_name, X_train, y_train, scoring):

    param_grids = {

        # --------------------------------------------------
        # RANDOM FOREST
        # --------------------------------------------------

        "RandomForestRegressor": {
            "model__n_estimators": [100, 200, 300],
            "model__max_depth": [None, 10, 20],
            "model__min_samples_split": [2, 5]
        },

        "RandomForestClassifier": {
            "model__n_estimators": [100, 200, 300],
            "model__max_depth": [None, 10, 20],
            "model__min_samples_split": [2, 5]
        },

        # --------------------------------------------------
        # XGBOOST
        # --------------------------------------------------

        "XGBRegressor": {
            "model__n_estimators": [200, 300, 400],
            "model__learning_rate": [0.01, 0.05, 0.1],
            "model__max_depth": [4, 6, 8]
        },

        "XGBClassifier": {
            "model__n_estimators": [200, 300, 400],
            "model__learning_rate": [0.01, 0.05, 0.1],
            "model__max_depth": [4, 6, 8]
        },

        # --------------------------------------------------
        # GRADIENT BOOSTING
        # --------------------------------------------------

        "GradientBoostingRegressor": {
            "model__n_estimators": [100, 200],
            "model__learning_rate": [0.05, 0.1],
            "model__max_depth": [3, 5]
        },

        "GradientBoostingClassifier": {
            "model__n_estimators": [100, 200],
            "model__learning_rate": [0.05, 0.1],
            "model__max_depth": [3, 5]
        }

    }

    # ------------------------------------------------------
    # If model has no tuning grid → train normally
    # ------------------------------------------------------

    if model_name not in param_grids:

        print("No tuning grid for this model. Using default parameters.")

        pipeline.fit(X_train, y_train)

        return pipeline

    # ------------------------------------------------------
    # Run Hyperparameter Search
    # ------------------------------------------------------

    print(f"\nRunning hyperparameter tuning for {model_name}...")

    search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_grids[model_name],
        n_iter=10,
        cv=5,
        scoring=scoring,
        n_jobs=-1,
        random_state=42
    )

    search.fit(X_train, y_train)

    print("Best parameters:", search.best_params_)

    return search.best_estimator_