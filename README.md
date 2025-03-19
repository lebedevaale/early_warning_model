# Early warning model

## Short Description
Three iterations of the predictive model development for the search of the avalanche-like behaviour of the stock exchange on the example of the volume of trading in the financial hourly data provided by Yahoo Finance and on the solar data from the Ace satellite from the L1 Lagrange point provided by the NASA.

## Papers that use these models
- "Self-organization of the stock exchange to the edge of a phase transition: empirical and theoretical studies", A. Dmitriev, A. Lebedev, V. Kornilov, V. Dmitriev, https://doi.org/10.3389/fphy.2024.1508465
- several more coming

## Data
- target - if the time series' MA would increase by n% and MV by m% in the next h hours, then the target is 1, otherwise 0
- predictors - physical measures from the complex systems analysis of the time series calculated with the moving window of 500 points
    - skewness
    - kurtosis
    - Hurst coefficient
    - Lyapunov exponent
    - correlation dimension
    - autocorrelation-at-lag-1
    - PSD
    - wavelet leaders first three cumulants
    - dynamics and variance of the mentioned physical measures calculated with the moving window of 8 points

## Metrics
- AUC
- Precision
- Recall
- F1-score
- KS-test (two sample)

## Iteration 1
Probit (and possibly logit model) that checks base usefullness of the variables. 

## Iteration 2
Descision forest, LightGBM, XGBoost and CatBoost with base hyperparameter optimization to check if more advanced models perform better. Variables' usefulness is checked with importnance.

## Iteration 3
LightGBM, XGBoost and CatBoost models with Optuna hyperparameter optimization and SHAP analysis of the feature importance.

## Structure of the repository
- files
    - config.cfg - configuration file with base params for the physical measures calculation
    - critical_transition_definition.py - research on the definition of the critical transition settings for the stock exchange and the solar data
    - EDA.ipynb - exploratory data analysis for the original datasets
    - functions.py - main functions for the project
    - import_and_preparation.ipynb - data import and preparation (including metrics calculation) for all of the time series
    - modelling.ipynb - main notebook for the modelling that contains all base results
- folders
    - Data - original and prepared datasets for the modelling
    - Definition_simulations - data from the definition of the critical transition research
    - Models - financial models' results of the 3rd iteration
    - Models_solar - solar models' results of the 3rd iteration
    - Params - feature importance results for the financial models of the 1st and 2nd iterations
    - Params_solar - feature importance results for the solar models of the 1st and 2nd iterations