# Base libraries
import copy
import operator
import numpy as np
import pandas as pd
from tqdm import tqdm

# Libraries for plotting
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Libraries for metrics'calculation
from PyEMD import EMD
import nolds
import pymultifracs.mfa as mfa
from pymultifracs.utils import build_q_log
import scipy as sp
from scipy.fft import fft

# Libraries for modelling
import networkx as nx
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
import tensorflow as tf
import statsmodels.api as sm
import sklearn.svm as svm
import sklearn.model_selection as modsel
import sklearn.ensemble as ens
import sklearn.metrics as metrics
from sklearn.inspection import permutation_importance as pi
from sklearn.feature_selection import mutual_info_regression

np.random.seed(0)

#---------------------------------------------------------------------------------------


def variables_dynamics(data:pd.DataFrame,
                       groupby:str,
                       mean_only:bool = False,
                       save:bool = False,
                       file_name:str = None) -> None:
    """
    Function for the plotting of the dynamics for the variables

    Inputs:
    --------------------
    data : pd.DataFrame
        Dataframe with columns for the analysis
    groupby : str
        Column to groupby
    mean_only : bool = False
        Whether to plot only means and not min-max
    save : bool = False
        Whether to save the plot
    file_name : str = None
        Name of the file to save the plot

    Prints:
    --------------------
    Dynamics of the variables
    """

     # Creating grid of subplots
    av_cols = data.drop(columns = [groupby]).columns
    fig = make_subplots(rows = len(av_cols), cols = 1, subplot_titles = [col for col in av_cols])

    # Calculating mean, min and max values of variables for each unique value in groupby column
    data_mean = data.groupby(groupby).mean()
    data_median = data.groupby(groupby).median()
    if mean_only != True:
        data_min = data.groupby(groupby).min()
        data_max = data.groupby(groupby).max()

    # Scattering returns
    for i, col in enumerate(av_cols):
        fig.add_trace(go.Scatter(x = data_mean.index, y = data_mean[col], mode = 'lines', name = f'{col}_mean', line = dict(color = 'green')), row = i + 1, col = 1)
        fig.add_trace(go.Scatter(x = data_mean.index, y = data_median[col], mode = 'lines', name = f'{col}_median', line = dict(color = 'purple')), row = i + 1, col = 1)
        if mean_only != True:
            fig.add_trace(go.Scatter(x = data_min.index, y = data_min[col], mode = 'lines', name = f'{col}_min', line = dict(color = 'red')), row = i + 1, col = 1)
            fig.add_trace(go.Scatter(x = data_max.index, y = data_max[col], mode = 'lines', name = f'{col}_max', line = dict(color = 'blue')), row = i + 1, col = 1)
        fig.update_xaxes(autorange = "reversed", row = i + 1, col = 1)
        fig.add_vline(x = 0, line_width = 3, line_dash = 'dash', line_color = 'black', row = i + 1, col = 1)

    # Update layout
    fig.update_layout(
        showlegend = False,
        font = dict(size = 20),
        height = 300 * len(av_cols),
        width = 1200
    )

    # Save the plot
    if save == True:
        fig.write_image(f'Plots/{file_name}.svg')

    # Show the plot
    fig.show()

#---------------------------------------------------------------------------------------

def heatmap(data:pd.DataFrame):
    """
    Function for the plotting of the correlation heatmap

    Inputs:
    --------------------
    data : pd.DataFrame
        Dataframe with columns for the analysis
    
    Prints:
    --------------------
    Correlation heatmap
    """

    # Creating grid of subplots
    fig = make_subplots(rows = 1, cols = 2, subplot_titles = ["Pearson Correlation", "Spearman Correlation"])

    # Add trace for each correlation matrix
    z1 = data.corr(method = 'pearson')
    z2 = data.corr(method = 'spearman')
    z = [z1, z2]
    for i in range(len(z)):
        fig.add_trace(go.Heatmap(z = z[i][::-1],
                                 x = data.columns,
                                 y = data.columns[::-1],
                                 text = z[i][::-1].round(2),
                                 texttemplate = "%{text}",
                                 zmin = -1, zmax = 1), 
                                 row = 1, col = i + 1)

    # Update layout
    fig.update_layout(
        showlegend = False,
        font = dict(size = 14),
        height = 600,
        width = 1600
    )
    fig.update_annotations(font_size = 30)

    # Show the plot
    fig.show()

#---------------------------------------------------------------------------------------

def roc_metric(Y:pd.DataFrame,
               Y_pred:pd.DataFrame) -> tuple:
    """
    Function for the calculation of AUC metric

    Inputs:
    ----------
    Y : DataFrame
        Set of Y for the model
    Y_pred : DataFrame
        Set of predicted Y for the model

    Returns:
    ----------
    auc : float
        AUC for the given series
    thresholds[optimal_index] : float
        Optimal threshold with highest (TPR - FPR)
    """
    
    fpr, tpr, thresholds = metrics.roc_curve(Y, Y_pred, pos_label=1)
    auc = round(metrics.auc(fpr, tpr), 3)
    optimal_index = np.argmax(tpr - fpr)

    return auc, thresholds[optimal_index]

#---------------------------------------------------------------------------------------

def remove_most_insignificant(X, 
                              X_test, 
                              results) -> tuple:
    """
    Function for the removal of the most insignificant variables from the model

    Inputs:
    ----------
    X : DataFrame
        Set of X for the model
    results : model
        Fitted statsmodels model

    Returns:
    ----------
    X : DataFrame
        Optimized set of X for the validation of the model
    X_test : DataFrame
        Optimized set of X for the testing of the model
    """
    
    # Use operator to find the key which belongs to the maximum value in the dictionary
    max_p_value = max(results.pvalues.items(), key = operator.itemgetter(1))[0]
    # Drop the worst feature
    X.drop(columns = max_p_value, inplace = True)
    X_test.drop(columns = max_p_value, inplace = True)

    return X, X_test

#---------------------------------------------------------------------------------------

def model_optimization(Y_train,
                       Y_val,
                       Y_test,
                       X_train,
                       X_val,
                       X_test,
                       type:str = 'Probit',
                       state:int = 0,
                       p_value_bord:float = 0.05, 
                       silent:bool = False,
                       insignificant_feature:bool = True,
                       target_metric:str = 'AUC') -> tuple:
    """
    Function for the optimization of OLS

    Inputs:
    ----------
    Y_train, Y_val, Y_test : array
        Target variable for the model
    X_train, X_val, X_test : DataFrame
        Set of X for the model
    type : str = 'Probit'
        Type of the model - 'Logit', 'Probit', 'RF', 'SVM', 'GB' or 'NN'
    state : int = 0
        Random state for the forest and SVM models
    p_value_bord : float = 0.05
        Maximum acceptable p-value for the coefficient
    silent : bool = False
        Whether not to show reports about model
    insignificant_feature : bool = True
        Whether to drop insignificant features or to keep them
    target_metric : str = 'AUC'
        Metric for the target, options: 'AUC', 'Precision

    Returns:
    ----------
    results : model
        Fitted statsmodels model
    [auc_train, auc_val, auc_test] : list[float, float, float]
        AUCs
    [ks_train.pvalue, ks_val.pvalue, ks_test.pvalue] : list[float, float, float]
        KS-test p-values
    [f1_train, f1_val, f1_test] : list[float, float, float]
        F1-scores
    [pr_train, pr_val, pr_test] : list[float, float, float]
        Precision scores
    [rec_train, rec_val, rec_test] : list[float, float, float]
        Recall scores
    """
    
    # Set a negative significance flag for forest models
    if type not in ['Probit', 'Logit']:
        insignificant_feature = False

    if insignificant_feature == False:
        # Create and fit model
        if type in ['Probit', 'Logit']:
            if type == 'Probit':
                model = sm.Probit(Y_train, X_train)
            else:
                model = sm.Logit(Y_train, X_train)
            results = model.fit(disp = 0)
        
        elif type in ['RF', 'SVM']:

            depth = 5

            if type == 'RF':
                model = ens.RandomForestClassifier(max_depth = depth, random_state = state, n_jobs = -1)
            else:
                model = svm.SVC(probability = True, random_state = state)
            model.fit(X_train, Y_train)

            # Get feature importance with feature permutation
            results = pi(model, X_test, Y_test, n_repeats = 10, random_state = state, n_jobs = -1)
        
        elif type in ['LightGBM', 'XGBoost', 'CatBoost']:

            metrics_mapper = {'LightGBM_AUC': 'auc', 'XGBoost_AUC': 'auc', 'CatBoost_AUC': 'AUC',
                              'LightGBM_Precision': 'average_precision', 'XGBoost_Precision': 'pre', 'CatBoost_Precision': 'Precision'}

            depth = 2

            if type == 'LightGBM':
                model = lgb.LGBMClassifier(max_depth = depth, metric = metrics_mapper[f'{type}_{target_metric}'], 
                                           importance_type = 'gain', min_data_in_leaf = 15,
                                           random_state = state, n_jobs = -1, verbosity = -1)
                model.fit(X_train, Y_train, eval_set = [(X_val, Y_val)],
                          callbacks = [lgb.early_stopping(100, verbose = False)])
                
                # Get feature importance
                results = model.feature_importances_
            
            elif type == 'XGBoost':
                model = xgb.XGBClassifier(max_depth = depth, eval_metric = metrics_mapper[f'{type}_{target_metric}'],
                                          early_stopping_rounds = 100, min_child_weight = 10, 
                                          random_state = state, verbosity = 0, n_jobs = -1, use_label_encoder = False)
                model.fit(X_train, Y_train, eval_set = [(X_val, Y_val)], verbose = False)

                # Get feature importance
                results = model.get_booster().get_score(importance_type = 'gain')
            
            else:
                model = CatBoostClassifier(depth = depth, eval_metric = metrics_mapper[f'{type}_{target_metric}'],
                                           random_state = state, verbose = 0)
                model.fit(X_train, Y_train, eval_set = [(X_val, Y_val)], early_stopping_rounds = 100, silent = True)
                
                # Get feature importance
                results = model.get_feature_importance()

        elif type == 'NN':
            # Initiate the model with specific layers
            model = tf.keras.models.Sequential([
                tf.keras.layers.Dense(10, activation = 'relu', input_shape = (len(X_train.columns),)),
                tf.keras.layers.Dense(1, activation = 'sigmoid')
            ])

            model.compile(optimizer = 'adam',
                loss = 'binary_crossentropy',
                metrics = ['AUC'])
            
            model.fit(X_train, Y_train, epochs=5)

            # Get feature importance
            results = None
        else:
            raise ValueError(f"Model type '{type}' is not supported.")

    while insignificant_feature:
        # Create model
        if type == 'Probit':
            model = sm.Probit(Y_train, X_train)
        else:
            model = sm.Logit(Y_train, X_train)

        # Fit model and get significant features
        results = model.fit(disp = 0)
        significant = [p_value < p_value_bord for p_value in results.pvalues]
        if all(significant):
            insignificant_feature = False
        else:
            # If there's only one insignificant variable left
            if X_train.shape[1] == 1:
                print('No significant features found')
                results = None
                insignificant_feature = False
            else:
                X_train, X_test = remove_most_insignificant(X_train, X_test, results)
                X_val = X_val[X_val.columns[X_val.columns.isin(X_train.columns)]]

    # Predictions and AUC
    if type in ['Logit', 'Probit']:
        Y_train_pred = results.predict(X_train)
        Y_val_pred = results.predict(X_val)
        Y_test_pred = results.predict(X_test)
    else:
        Y_train_pred = model.predict_proba(X_train)[:, 1]
        Y_val_pred = model.predict_proba(X_val)[:, 1]
        Y_test_pred = model.predict_proba(X_test)[:, 1]
    auc_train, threshold_train = roc_metric(Y_train, Y_train_pred)
    auc_val, threshold_val = roc_metric(Y_val, Y_val_pred)
    auc_test, threshold_test = roc_metric(Y_test, Y_test_pred)
    Y_train_pred_round = np.where(Y_train_pred < threshold_train, np.floor(Y_train_pred), np.ceil(Y_train_pred))
    Y_val_pred_round = np.where(Y_val_pred < threshold_val, np.floor(Y_val_pred), np.ceil(Y_val_pred))
    Y_test_pred_round = np.where(Y_test_pred < threshold_test, np.floor(Y_test_pred), np.ceil(Y_test_pred))

    # KS-test
    ks_samples_train = pd.DataFrame({'Y': Y_train, 'Y_pred': Y_train_pred})
    ks_samples_train_posi = ks_samples_train[ks_samples_train['Y'] == 1]['Y_pred']
    ks_samples_train_nega = ks_samples_train[ks_samples_train['Y'] == 0]['Y_pred']
    ks_train = sp.stats.kstest(ks_samples_train_posi, ks_samples_train_nega)
    ks_samples_val = pd.DataFrame({'Y': Y_val, 'Y_pred': Y_val_pred})
    ks_samples_val_posi = ks_samples_val[ks_samples_val['Y'] == 1]['Y_pred']
    ks_samples_val_nega = ks_samples_val[ks_samples_val['Y'] == 0]['Y_pred']
    ks_val = sp.stats.kstest(ks_samples_val_posi, ks_samples_val_nega)
    ks_samples_test = pd.DataFrame({'Y': Y_test, 'Y_pred': Y_test_pred})
    ks_samples_test_posi = ks_samples_test[ks_samples_test['Y'] == 1]['Y_pred']
    ks_samples_test_nega = ks_samples_test[ks_samples_test['Y'] == 0]['Y_pred']
    ks_test = sp.stats.kstest(ks_samples_test_posi, ks_samples_test_nega)

    # F1-score, precision and recall
    f1_train = round(metrics.f1_score(Y_train, Y_train_pred_round), 3)
    f1_val = round(metrics.f1_score(Y_val, Y_val_pred_round), 3)
    f1_test = round(metrics.f1_score(Y_test, Y_test_pred_round), 3)
    pr_train = round(metrics.precision_score(Y_train, Y_train_pred_round), 3)
    pr_val = round(metrics.precision_score(Y_val, Y_val_pred_round), 3)
    pr_test = round(metrics.precision_score(Y_test, Y_test_pred_round), 3)
    rec_train = round(metrics.recall_score(Y_train, Y_train_pred_round), 3)
    rec_val = round(metrics.recall_score(Y_val, Y_val_pred_round), 3)
    rec_test = round(metrics.recall_score(Y_test, Y_test_pred_round), 3)
    if silent == False:
        print(f'''Train AUC score: {auc_train}, Train KS-test p-value: {round(ks_train.pvalue, 3)}, 
              Train F1-score: {f1_train}, Train precision: {pr_train}, Train recall: {rec_train}''')
        print(f'''Validation AUC score: {auc_test}, Validation KS-test p-value: {round(ks_test.pvalue, 3)}, 
              Validation F1-score: {f1_test}, Validation precision: {pr_test}, Validation recall: {rec_test}''')
        print(f'''Test AUC score: {auc_test}, Test KS-test p-value: {round(ks_test.pvalue, 3)}, 
              Test F1-score: {f1_test}, Test precision: {pr_test}, Test recall: {rec_test}''')
        if type in ['Logit', 'Probit']:
            print(results.summary())

    return results, [auc_train, auc_val, auc_test],\
           [round(ks_train.pvalue, 9), round(ks_val.pvalue, 9), round(ks_test.pvalue, 9)],\
           [f1_train, f1_val, f1_test], [pr_train, pr_val, pr_test],\
           [rec_train, rec_val, rec_test]

#---------------------------------------------------------------------------------------

def model(data,
          target:str,
          horizons:list,
          shares:list,
          states:list,
          model:str = 'Probit',
          separate:bool = False,
          target_metric:str = 'AUC') -> pd.DataFrame:
    """
    Function for the Monte Carlo simulation of the samples and modelling

    Inputs:
    --------------------
    data : pd.DataFrame
        Dataframe with data for modelling
    target : str
        Name of the target column
    horizons : list
        List of possible horizons
    shares : list
        List of possible shares of target equal to 1 in the dataset
    states : list
        List of random states
    model : str = 'Probit'
        Model type: 'Logit', 'Probit', 'RF', 'SVM' or 'GB
    separate : bool = False
        Whether to calculate whole logit or probit models or to separate variables to different models
    target_metric : str = 'AUC'
        Metric for the target, options: 'AUC', 'Precision

    Returns:
    --------------------
    res : pd.DataFrame
        Dataframe with raw statistical results of the modelling
    """

    def balance_and_split(data_testing_0, Y_1, share_1_orig, target, share, state, model = 'Probit'):
        
        # Balance the negative and positive samples if needed
        if share != None:
            # Drop part of the negative samples to balance sample
            _, X_0, _, Y_0 = modsel.train_test_split(data_testing_0.drop(columns = [target]), data_testing_0[target], 
                                                     test_size = min(share_1_orig * (1 - share) / share, 1), random_state = state)
            share_1 = len(Y_1) / (len(Y_0) + len(Y_1))
            Y = pd.concat([Y_0, Y_1])
            if model in ['Logit', 'Probit']:
                X = sm.add_constant(pd.concat([X_0, X_1]))

        # Split the data into train, validation and test
        X_train, X_test, Y_train, Y_test = modsel.train_test_split(X, Y, test_size = 0.2, random_state = state)
        X_train, X_val, Y_train, Y_val = modsel.train_test_split(X_train, Y_train, test_size = 0.16, random_state = state)
        
        return X_train, X_val, X_test, Y_train, Y_val, Y_test, share_1

    # Define columns for the dataframe
    columns = ['Horizon', '1 Share', '1 Share real', 'State',
               'Train size', 'Validation size', 'Test size',
               'Train AUC', 'Validation AUC', 'Test AUC',
               'Train KS-test p-value', 'Validation KS-test p-value', 'Test KS-test p-value',
               'Train F1-score', 'Validation F1-score', 'Test F1-score', 
               'Train precision', 'Validation precision', 'Test precision', 
               'Train recall', 'Validation recall', 'Test recall'] 
    
    # Update set of columns for the dataframe
    if model in ['Logit', 'Probit']:
        columns += ['Coeffs']
        if separate == True:
            columns = columns[:4] + ['Variable'] + columns[4:] + ['Pvalues']
    else:
        columns += ['Importance']

    # Create dataframe for the results
    res = pd.DataFrame(columns = columns)

    # Iterate over the chosen parameters and optimize classification models, then save all the results to the dataframe
    for horizon in tqdm(horizons):
        data_testing = data.copy()
        data_testing['Flag'] = data_testing['Distance'].apply(lambda x: 0 if x > horizon else 1)
        data_testing.drop(columns = ['Volume', 'MA100', 'MV100', 'Rise', 'Distance', 'Index', 'Ticker'], inplace = True)
        
        # Split the data to positive and negative samples
        data_testing_1 = data_testing[data_testing[target] == 1]
        data_testing_0 = data_testing[data_testing[target] == 0]
        Y_1 = data_testing_1[target]
        
        if model in ['Logit', 'Probit']:
            # Choose one of the two models
            if separate == False:
                X_1 = data_testing_1.drop(columns = [target])
                share_1_orig = len(data_testing_1) / (len(data_testing_0) + len(data_testing_1))
                for share in shares:
                    for state in states:

                        # Balance classes and split the data into train and test
                        X_train, X_val, X_test, Y_train, Y_val, Y_test, share_1 = balance_and_split(data_testing_0, Y_1, share_1_orig,
                                                                                                    target,share, state)
                        
                        # Calculate models and metrics
                        results_rs, auc_rs, ks_rs, f1_rs, pr_rs, rec_rs,\
                            = model_optimization(Y_train, Y_val, Y_test, X_train, X_val, X_test, type = model, 
                                                 silent = True, target_metric = target_metric)
                        res.loc[len(res)] = [horizon, share, share_1, state,
                                             len(Y_train), len(Y_val), len(Y_test),
                                             *auc_rs, *ks_rs, *f1_rs, *pr_rs, *rec_rs,
                                             results_rs.params]
            else:
                for col in data_testing.columns.drop(target):
                    X_1 = data_testing_1[col]
                    share_1_orig = len(data_testing_1) / (len(data_testing_0) + len(data_testing_1))
                    for share in shares:
                        for state in states:

                            # Balance classes and split the data into train and test
                            X_train, X_val, X_test, Y_train, Y_val, Y_test, share_1 = balance_and_split(data_testing_0, Y_1, share_1_orig,
                                                                                                        target, share, state)
                            
                            # Calculate models and metrics
                            try:
                                results_rs, results_rs, auc_rs, ks_rs, f1_rs, pr_rs, rec_rs,\
                                    = model_optimization(Y_train, Y_val, Y_test, X_train, X_val, X_test, type = model, 
                                                        silent = True, insignificant_feature = False, target_metric = target_metric)
                                res.loc[len(res)] = [horizon, share, share_1, state, col,
                                                     len(Y_train), len(Y_val), len(Y_test),
                                                     *auc_rs, *ks_rs, *f1_rs, *pr_rs, *rec_rs, 
                                                     results_rs.params, results_rs.pvalues]
                            except:
                                pass
        
        elif model in ['RF', 'SVM', 'LightGBM', 'XGBoost', 'CatBoost']:
            X_1 = data_testing_1.drop(columns = [target])
            share_1_orig = len(data_testing_1) / (len(data_testing_0) + len(data_testing_1))
            for share in shares:
                for state in states:

                    # Balance classes and split the data into train and test
                    X_train, X_val, X_test, Y_train, Y_val, Y_test, share_1 = balance_and_split(data_testing_0, Y_1, share_1_orig,
                                                                                                target, share, state)

                    # Calculate models and metrics
                    results_rs, auc_rs, ks_rs, f1_rs, pr_rs, rec_rs,\
                        = model_optimization(Y_train, Y_val, Y_test, X_train, X_val, X_test, type = model,
                                             silent = True, target_metric = target_metric)
                    
                    if model in ['RF', 'SVM']:
                        res.loc[len(res)] = [horizon, share, share_1, state,
                                             len(Y_train), len(Y_val), len(Y_test),
                                             *auc_rs, *ks_rs, *f1_rs, *pr_rs, *rec_rs, 
                                             pd.Series(results_rs.importances_mean, index = X_train.columns.values)]
                    else:
                        res.loc[len(res)] = [horizon, share, share_1, state,
                                             len(Y_train), len(Y_val), len(Y_test),
                                             *auc_rs, *ks_rs, *f1_rs, *pr_rs, *rec_rs, 
                                             pd.Series(results_rs, index = X_train.columns.values)]
        else:
            raise ValueError('Unknown model type')
        
    return res  

#---------------------------------------------------------------------------------------

def save_results(res:pd.DataFrame,
                 cols:list,
                 model:str = 'Probit',
                 sep:bool = False,
                 path:str = 'Params') -> pd.DataFrame:
    """
    Function for the saving of the simulation results
    
    Inputs:
    ---------
    res : pd.DataFrame
        Results from simulations
    cols:list
        Columns for the transformation
    model : str = 'Probit'
        Model type - 'Logit', 'Probit', 'RF', 'SVM' or 'GB'
    sep : bool = 'False'
        Whether the original models were separate Logit or Probit
    path : str
        Path to save the results
    
    Returns:
    ---------
    res_means : pd.DataFrame
        Pivot for the results
    """
    
    # Define where the data that needs to be transformed is stored
    if model in ['Logit', 'Probit']:
        info = 'Coeffs'
    elif model in ['RF', 'SVM', 'LightGBM', 'XGBoost', 'CatBoost']:
        info = 'Importance'
        sep = False
    else:
        raise ValueError('Unknown model type')
    
    # Define the additional flag of the model
    if sep == True:
        sep_name = '_sep'
    else:
        sep_name = ''

    # OHE-like transformation of the variables' lists
    if sep == False:
        res_coeffs = pd.DataFrame(columns = list(cols))
        for row in res[info]:
            res_coeffs.loc[len(res_coeffs)] = row
        res = res.drop(columns = [info]).join(res_coeffs)
        res.to_parquet(f'{path}/params_{model}.parquet')
        groups = ['Horizon', '1 Share', '1 Share real']
        drops = ['State']
    else:
        res['Const'] = res['Coeffs'].apply(lambda x: x['const'].item())
        res['Const_Pvalue'] = res['Pvalues'].apply(lambda x: x['const'].item())
        coef = []
        coef_p = []
        for row in res.itertuples():
            coef.append(row.Coeffs[row.Variable])
            coef_p.append(row.Pvalues[row.Variable])
        res['Coef'] = coef
        res['Coef_Pvalue'] = coef_p
        res.drop(columns = ['Coeffs', 'Pvalues']).to_parquet(f'{path}/params_sep_{model}.parquet')
        groups = ['Variable', 'Horizon']
        drops = ['State', 'Coeffs', '1 Share', '1 Share real']
    
    # Create pivot based on the horizon and 1 share parameters
    res_means = res.groupby(groups)[res.columns.drop(groups + drops)].mean()
    res_means.to_parquet(f'{path}/params{sep_name}_mean_{model}.parquet')

    return res_means

#---------------------------------------------------------------------------------------
    
def generate_random_series(length:int, 
                           number:int, 
                           mean:float = None, 
                           sigma:float = None, 
                           type:str = 'normal') -> pd.DataFrame:
    """
    Function for the generation of random series
    
    Inputs:
    ---------
    length : int
        Length of the series
    number : int
        Number of the series
    mean : float
        Mean of the distribution
    sigma : float
        Sigma of the distribution
    type : str = 'normal'
        Type of the series, may be 'normal', 'lognormal' or 'rw'
    
    Returns:
    ---------
    res : pd.DataFrame

    Raises:
    ---------
    ValueError: If type of the series is not 'normal', 'lognormal' or 'rw'
    """

    # Normal distribution
    if type == 'normal':
        res = pd.DataFrame(np.random.normal(loc = mean, scale = sigma, size = (length, number)), 
                           columns = [str(i) for i in range(number)])
    
    # Lognormal distribution
    elif type == 'lognormal':
        res = pd.DataFrame(np.random.lognormal(mean = mean, sigma = sigma, size = (length, number)), 
                           columns = [str(i) for i in range(number)])
    # Random walk
    elif type == 'rw':
        res = pd.DataFrame(np.cumsum(np.random.randn(length, number), axis = 1), columns = [str(i) for i in range(number)])

    else:
        raise ValueError('Incorrect type of the series')
    
    return res

#---------------------------------------------------------------------------------------

def graph_generation(graph_type:str,
                     number_of_nodes:int, 
                     BA_connect:int = None, 
                     ER_prob:float = None,
                     CL_average:int = None,
                     silent:bool = True) -> nx.Graph:
    """
    Function that generates a graph based on the specified graph type.

    Inputs:
    ---------
    graph_type : str
        The type of graph to generate. Valid values are 'BA', 'ER', and 'CL'.
    number_of_nodes : int
        The number of nodes in the graph.
    BA_connect : int = None
        The number of connections in the Barabasi-Albert graph. Required if graph_type is 'BA'.
    ER_prob : float = None
        The probability of an edge in the Erdos-Renyi graph. Required if graph_type is 'ER'.
    CL_average : int = None
        The average degree in the Chung-Lu graph. Required if graph_type is 'CL'.
    silent : bool = True
        If True, suppresses the print of graph size (number of edges).

    Returns:
    ---------
    G : nx.Graph
        The generated graph.

    Raises:
    ---------
    ValueError: If the graph_type is not one of 'BA', 'ER', or 'CL'.
    """
    
    if graph_type == 'BA':
        G = nx.barabasi_albert_graph(number_of_nodes, BA_connect)
    elif graph_type == 'ER':
        G = nx.erdos_renyi_graph(number_of_nodes, ER_prob)
    elif graph_type == 'CL':
    # Model's Parameters: Generate a random Chung-Lu graph with average degree d, max degree m, and power-law degree distribution with exponent gamma
    # Source: https://github.com/ftudisco/scalefreechunglu/blob/master/python/example.py
        gamma = 2.2
        m = number_of_nodes ** 0.4
        p = 1 / (gamma - 1)
        c = (1 - p) * CL_average * (number_of_nodes ** p)
        i0 = (c / m) ** (1 / p) - 1
        w = [c / ((i + i0) ** p) for i in range(number_of_nodes)]
        G = nx.expected_degree_graph(w)
    else:
        raise ValueError("Wrong graph_type. Choose BA, ER or CL.")
    
    if silent == False: 
        print(G.size())

    return G

#---------------------------------------------------------------------------------------

def grain_generator(number_of_nodes:int, 
                    number_of_days:int, 
                    dist:str = 'uni') -> list:
    """
    Generate grains to be put into the sand pile based on the specified distribution for a given number of nodes and days.

    Inputs:
    ---------
    number_of_nodes : int
        The number of nodes to generate grains for.
    number_of_days : int
        The number of days for which grains are generated.
    dist : str = 'uni'
        The type of distribution to use for generating grains. Default is 'uni'.
        Other options: 'expon' for exponential, 'par' for Pareto.

    Returns:
    ---------
    new_grains : list
        A list of generated grains based on the specified distribution.

    Raises:
    ---------
    ValueError: If the distribution type is not one of 'uni', 'expon' or 'par'.
    """
    
    # Create list to fday
    new_grains = []
    
    # Generate grains depending on the distribution
    if dist == 'uni':
        for d in range(number_of_days):
            grain = np.random.randint(number_of_nodes)
            new_grains.append([grain])
    elif dist in ['expon', 'par']:
        if dist == 'expon':
            num_of_grains = np.around(np.random.exponential(size = number_of_days))
        else:
            num_of_grains = np.around(np.random.pareto(a = 2, size = number_of_days))
        for d in range(number_of_days):
            grains = []
            for g in range(int(num_of_grains[d]) + 1):
                grain = np.random.randint(number_of_nodes)
                grains.append(grain)
            new_grains.append(grains)
    else:
        raise ValueError('Wrong distribution type. Choose uni, expon or par.')
    
    return new_grains

#---------------------------------------------------------------------------------------

def spread_model(G:nx.Graph, 
                 ones:list, 
                 falls_d:int, 
                 d:int, 
                 node:list, 
                 crit:int, 
                 type:str = 'BTW',
                 facilit_list:list = None) -> tuple:
    """
    Function for the implementation of different spread models on the graph.
    Code for the facilitated models should be optimised further, but I gave up on it.
    
    Inputs:
    ---------
    G : nx.Graph 
        The graph on which the fall model is applied.
    ones : list 
        List of nodes with only one edge.
    falls_d : int 
        The number of iterations.
    d : int
        The current day.
    node : list
        The node to be processed.
    crit : int
        The critical value.
    type : str = 'BTW'
        The type of model to be used. Default is 'BTW'. Other option: 'MA'.
    facilit_list : list = None
        List of nodes to facilitate the spread.
    
    Returns:
    ---------
        tuple: A tuple containing the updated graph G and the updated number of iterations falls_d.
    
    Raises:
    ---------
    ValueError: If the type of model is not one of 'BTW' or 'MA'.
    """

    # Increase the number of iteration
    falls_d += 1

    # Iterate over nodes that have more than one edge
    if facilit_list == None:
        if node[0] not in ones:
            neighbors = [n for n in G.neighbors(node[0])]
            remains = copy.copy(crit)
            for neighbor in G.nodes(data=True):
                if neighbor[0] in neighbors:
                    # Spread the grains over the neighbors
                    if type == 'BTW':
                        # Deterministic model
                        neighbor[1]['day'+str(d+1)] += 1
                    elif type == 'MA':
                        # Stochastic model
                        n = np.random.randint(0, remains)
                        remains -= n
                        neighbor[1]['day'+str(d+1)] += n
                    else:
                        raise ValueError('Wrong model type. Choose BTW or MA.')
        
        # Update the number of grains in the node
        node[1]['day'+str(d+1)] -= crit
    
    else:
        # Deterministic model
        if type == 'BTW':
            if node[0] not in ones:
                neighbors = [n for n in G.neighbors(node[0])]

                # Spread the grains over the neighbors from not less than critical node
                if node[1]['day'+str(d)] >= crit:
                    for neighbor in G.nodes(data=True):
                        if neighbor[0] in neighbors:
                            neighbor[1]['day'+str(d+1)] += 1
                            facilit_list[d+1][neighbor[0]] += 1
                    node[1]['day'+str(d+1)] -= crit

                # Spread the grains over the neighbors from the facilitated node
                elif node[1]['day'+str(d)] < crit and node[1]['day'+str(d)] > 0:
                    remains = node[1]['day'+str(d)]
                    for neighbor in G.nodes(data=True):
                        if neighbor[0] in neighbors:
                            n = np.random.randint(0, remains)
                            remains -= n
                            neighbor[1]['day'+str(d+1)] += n
                            if n > 0:
                                facilit_list[d+1][neighbor[0]] += 1
                    node[1]['day'+str(d+1)] -= node[1]['day'+str(d)]
            else:
                node[1]['day'+str(d+1)] -= crit
        
        # Stochastic model
        elif type == 'MA':
            if node[0] not in ones:
                neighbors = [n for n in G.neighbors(node[0])]

                # Spread the grains over the neighbors from not less than critical node
                if node[1]['day'+str(d)] >= crit:
                    remains = copy.copy(crit)
                    for neighbor in G.nodes(data=True):
                        if neighbor[0] in neighbors:
                            n = np.random.randint(0, remains)
                            remains -= n
                            neighbor[1]['day'+str(d+1)] += n
                            if n > 0 and d+1 <= len(facilit_list):
                                facilit_list[d+1][neighbor[0]] += 1
                    node[1]['day'+str(d+1)] -= crit

                # Spread the grains over the neighbors from the facilitated node
                elif node[1]['day'+str(d)] < crit and node[1]['day'+str(d)] > 0:
                    remains = node[1]['day'+str(d)]
                    for neighbor in G.nodes(data=True):
                        if neighbor[0] in neighbors:
                            n = np.random.randint(0, remains)
                            remains -= n
                            neighbor[1]['day'+str(d+1)] += n
                            if n > 0 and d+1 <= len(facilit_list):
                                facilit_list[d+1][neighbor[0]] += 1
                    node[1]['day'+str(d+1)] -= node[1]['day'+str(d)]
            else:
                node[1]['day'+str(d+1)] -= crit
    
    return G, falls_d, facilit_list

#---------------------------------------------------------------------------------------

def spread(model:str,
           G:nx.Graph,
           number_of_days:int,
           new_grains:list,
           facilitated:bool = False,
           ad_dissipation:bool = False,
           neutral_state:bool = False,
           new_grains_plus:list = None,
           new_grains_minus: list = None,
           silent:bool = True) -> list:
    """
    Function for modelling of sand grain spread with Bak-Tang-Wiesenfeld and Manna on random graphs.

    Inputs:
    ---------
    model : str 
        The type of spread model to be used.
    G : nx.Graph 
        The graph on which the spread model is applied.
    number_of_days : int
        The number of days to simulate the spread.
    new_grains : list
        A list of lists, where each inner list contains the nodes where new grains are added on each day.
    facilitated : bool = False
        If True, the spread model includes spread based on the previous falls.
    ad_dissipation : bool
        If True, the spread model includes ad dissipation (+2-1 instead of +1-0).
    neutral_state : bool
        If True, the spread model includes neutral state (+1-1 instead of +1-0).
    new_grains_plus : list
        A list of lists, where each inner list contains the nodes where new grains are added on each day.
    new_grains_minus : list
        A list of lists, where each inner list contains the nodes where new grains are subtracted on each day.

    Returns:
    ---------
    falls : list 
        A list containing the number of falls for each day.
    """

    # Initialize variables
    falls = []
    ones = []
    deg = []

    # Get the degree of each node
    degrees = [[node, val]  for (node, val) in G.degree()]
    for degree in degrees:
        deg.append(degree[1])
        if degree[1] == 1:
            ones.append(degree[0])

    # Create status dataframe
    status = pd.DataFrame()
    for j in range(number_of_days):
        status['day' + str(j)] = np.zeros(G.number_of_nodes())

    # Set node attributes
    node_attr = status.to_dict('index')
    nx.set_node_attributes(G, node_attr)

    # Create a list of nodes to facilitate the spread if needed
    if facilitated == True:
        facilit_list = [[0 for x in range(G.number_of_nodes())] for z in range(number_of_days)]
    else:
        facilit_list = None

    # Simulate the spread
    for d in tqdm(range(0, number_of_days-1), disable = silent):
        falls_d = 0

        # Add/subtract grains based on the model
        if ad_dissipation == True:
            for node in G.nodes(data = True):
                if (node[0] in new_grains[d]) | (node[0] in new_grains_plus[d]):
                    node[1]['day' + str(d)] += 1.0
                if node[0] in new_grains_minus[d]:
                    node[1]['day' + str(d)] -= 1.0          
                node[1]['day' + str(d + 1)] += node[1]['day' + str(d)]
        elif neutral_state == True:
            for node in G.nodes(data = True):
                if node[0] in new_grains[d]:
                    node[1]['day' + str(d)] += 1.0
                if node[0] in new_grains_minus[d]:
                    node[1]['day' + str(d)] -= 1.0          
                node[1]['day' + str(d + 1)] += node[1]['day' + str(d)]    
        else:
            for node in G.nodes(data = True):
                if node[0] in new_grains[d]:
                    node[1]['day' + str(d)] += 1.0
                node[1]['day' + str(d + 1)] += node[1]['day' + str(d)]

        # Spread grains for the base or for the facilitated model
        for node in G.nodes(data = True):
            if facilit_list == None:
                if (d <= (number_of_days - 1)) and (node[1]['day' + str(d)] >= deg[node[0]]) and (deg[node[0]]) > 0:
                    G, falls_d, _ = spread_model(G, ones, falls_d, d, node, deg[node[0]], type = model)
            else:
                if (d <= (number_of_days - 1)) and ((node[1]['day' + str(d)] >= deg[node[0]]) or (facilit_list[d][node[0]] >= 2)) and (deg[node[0]]) > 0:
                    G, falls_d, facilit_list = spread_model(G, ones, falls_d, d, node, deg[node[0]], type = model, facilit_list = facilit_list)

        # Append number of falls
        if d <= number_of_days-2:
            falls.append(falls_d)

    return falls

#---------------------------------------------------------------------------------------

def critical_transition(data:pd.DataFrame,
                        crit:int,
                        window:int) -> pd.DataFrame:
    """
    Function for the calculation of the critical transition.

    Inputs:
    ---------
    data : pd.DataFrame
        Dataframe with columns for the analysis
    crit : int
        The critical value.
    window : str
        The window size.

    Returns:
    ---------
    data : pd.DataFrame
    """

    # Calculate the critical transition
    for col in data.columns:
        data[f'{col}, MA'] = data[col].rolling(int(window)).mean()
        data[f'{col}, Var'] = data[col].rolling(int(window)).var()
        data[f'{col}, Dynamics MA'] = data[f'{col}, MA'] / data[f'{col}, MA'].shift(5)
        data[f'{col}, Dynamics Var'] = data[f'{col}, Var'] / data[f'{col}, Var'].shift(5)
        data[f'{col}, Rise'] = (data[f'{col}, Dynamics MA'] > crit) & (data[f'{col}, Dynamics Var'] > crit)

    return data

#---------------------------------------------------------------------------------------------------------------------------------------

def emd(signal, 
        t, 
        plot:bool = False):
    """
    Function for the decomposition of time series to the several components until the last one is monotonous
    Source: https://towardsdatascience.com/improve-your-time-series-analysis-with-stochastic-and-deterministic-components-decomposition-464e623f8270
    
    Inputs:
    ----------
    signal : array
        Time series for decomposition
    t : array
        Index of time series for plotting
    plot : bool = False
        Flag whether to plot of decomposed time series is needed

    Plots:
    ----------
    Plots of original time series and its decomposed parts if plot == True

    Returns:
    ----------
    imfs : array
        Decomposed time series
    """

    # Separate time series into components
    emd = EMD(DTYPE = np.float16, spline_kind = 'akima')
    imfs = emd(signal.values)
    N = imfs.shape[0]
    
    if plot:
        # Creating grid of subplots
        fig = make_subplots(rows = N + 1, cols = 1, subplot_titles = ['Original Signal'] + [f'IMF {i}' for i in range(N)])

        # Scattering signal and IMFs
        fig.add_trace(go.Scatter(x = t, y = signal, mode = 'lines', name = 'Original Signal'), row = 1, col = 1)
        for i, imf in enumerate(imfs):
            fig.add_trace(go.Scatter(x = t, y = imf, mode = 'lines', name = f'IMF {i}'), row = i + 2, col = 1)

        # Update layout
        fig.update_layout(
            showlegend = False,
            font = dict(size = 20),
            height = 400 * (N + 1),
            width = 2000
        )
        fig.show()

    return imfs

#---------------------------------------------------------------------------------------------------------------------------------------

def phase_spectrum(imfs):
    """
    Function for the calculation of the time series' phase spectrum
    Source: https://towardsdatascience.com/improve-your-time-series-analysis-with-stochastic-and-deterministic-components-decomposition-464e623f8270
    
    Inputs:
    ----------
    imfs : array
        Decomposed time series

    Returns:
    ----------
    imfs_p : array
        Phase spectrum of decomposed time series
    """

    # Iterate over decomposed timer series to calculate each ones phase spectrum
    imfs_p = []
    for imf in imfs:
        trans = fft(imf)
        imf_p = np.arctan(trans.imag / trans.real)
        imfs_p.append(imf_p)

    return imfs_p

#---------------------------------------------------------------------------------------------------------------------------------------

def phase_mi(phases):
    """
    Function for the calculation of mutual information in the phases
    Source: https://towardsdatascience.com/improve-your-time-series-analysis-with-stochastic-and-deterministic-components-decomposition-464e623f8270
    
    Inputs:
    ----------
    phases : array
        Phase spectrum of decomposed time series
    
    Returns:
    ----------
    mis : array
        Mutual information of phase spectrums of decomposed time series
    """

    # Iterate over phases to calculate mutual info
    mis = []
    for i in range(len(phases) - 1):
        mis.append(mutual_info_regression(phases[i].reshape(-1, 1), phases[i + 1])[0])
        
    return np.array(mis)

#---------------------------------------------------------------------------------------------------------------------------------------

def divide_signal(signal, 
                  t, 
                  imfs, 
                  mis, 
                  cutoff:float = 0.05, 
                  plot = False):
    """
    Function for the final separation to the stohastic and determenistic components
    Source: https://towardsdatascience.com/improve-your-time-series-analysis-with-stochastic-and-deterministic-components-decomposition-464e623f8270
    
    Inputs:
    ----------
    signal : array
        Time series for decomposition
    t : array
        Index of time series for plotting
    imfs : array
        Decomposed time series
    mis : array
        Mutual information of phase spectrums of decomposed time series
    cutoff : float = 0.05
        Border of separation between stohastic and determenistic components
    plot : bool = False
        Flag whether to plot original time series, stohastic and determenistic components

    Plots:
    ----------
    Plots of original time series, stohastic and determenistic components if plot == True

    Returns:
    ----------
    stochastic_component : array
        Sum of time series components that are considered stohastic
    deterministic_component : array
        Sum of time series components that are considered deterministic
    """

    # Separate time series to stohastic and deterministic components 
    cut_point = np.where(mis > cutoff)[0][0]    
    stochastic_component = np.sum(imfs[:cut_point], axis=0)
    deterministic_component = np.sum(imfs[cut_point:], axis=0)

    if plot:
        # Creating grid of subplots
        fig = make_subplots(rows = 3, cols = 1, subplot_titles = ['Original Signal', 'Stochastic Component', 'Deterministic Component'])
        
        # Scattering signal and components
        fig.add_trace(go.Scatter(x = t, y = signal, mode = 'lines', name = 'Original Signal'), row = 1, col = 1)
        fig.add_trace(go.Scatter(x = t, y = stochastic_component, mode = 'lines', name = 'Stochastic Component'), row = 2, col = 1)
        fig.add_trace(go.Scatter(x = t, y = deterministic_component, mode = 'lines', name = 'Deterministic Component'), row = 3, col = 1)

        # Update layout
        fig.update_layout(
            showlegend = False,
            font = dict(size = 20),
            height = 1200,
            width = 2000
        )
        fig.show()
    
    return stochastic_component, deterministic_component

#---------------------------------------------------------------------------------------------------------------------------------------

def find_rise(data:pd.DataFrame,
              column:str,
              ma:float,
              mv:float,
              window:int) -> pd.DataFrame:
    """
    Function for the calculation of critical transitions in the time series

    Inputs:
    ----------
    data : pd.DataFrame
        Dataframe with data for the analysis
    column : str
        Column for the analysis
    ma : float
        Critical value for MA
    mv : float
        Critical value for MV
    window : str
        The window size for MA and MV

    Returns:
    ----------
    data_final : pd.DataFrame
        Dataframe with calculated values
    """
    # Get column for the analysis
    data_final = pd.DataFrame(data.index).set_index(0)
    data_final = data[column].to_frame().dropna()

    # Calculate MA and MV
    data_final[f'MA{window}'] = data_final[column].rolling(int(window)).mean()
    data_final[f'MV{window}'] = data_final[column].rolling(int(window)).var()

    # Calculate dynamics for MA and MV
    data_final['Dynamics MA'] = data_final[f'MA{window}'] / data_final[f'MA{window}'].shift(5)
    data_final['Dynamics MV'] = data_final[f'MV{window}'] / data_final[f'MV{window}'].shift(5)

    # Find critical transitions
    data_final['Rise'] = (data_final['Dynamics MA'] >= ma) & (data_final['Dynamics MV'] >= mv)
    return data_final

#---------------------------------------------------------------------------------------------------------------------------------------

def find_transitions(data_final:pd.DataFrame,
                     ma:float,
                     mv:float,
                     window:int,
                     distance:int,
                     tail:int):
    """
    Function for the calculation of critical transitions in the time series

    Inputs:
    ----------
    data_final : pd.DataFrame
        Dataframe with data for the analysis
    ma : float
        Critical value for MA
    mv : float
        Critical value for MV
    window : str
        The window size for MA and MV
    distance : int
        Minimal distance between critical transitions
    tail : int
        Tail of critical transitions that will separate it from noise outliers

    Returns:
    ----------
    data_transitions : pd.DataFrame
        Dataframe with calculated values
    """

    # Separate critical transitions from the dataset
    data_transitions = data_final[data_final['Rise'] == True]
    transitions = data_transitions.index
    final_transitions = []

    # Find transitions that have the sufficient distance between them and previous ones and have a tail
    for i, index in enumerate(transitions):
        if i < len(transitions) - 1:
            if (index >= distance) & (index - transitions[i - 1] > distance + tail):
               if (data_final[f'MA{window}'][index + tail].item() / data_final[f'MA{window}'][index - 5].item() > ma) &\
                (data_final[f'MV{window}'][index + tail].item() / data_final[f'MV{window}'][index - 5].item() > mv):
                    final_transitions.append(index)
    return data_transitions[data_transitions.index.isin(final_transitions)]

#---------------------------------------------------------------------------------------------------------------------------------------

def get_metrics(data:pd.DataFrame,
                active_tickers:list,
                ma:float,
                mv:float,
                window:int,
                distance:int,
                tail:int,
                window_metrics:int) -> pd.DataFrame:
    """
    Function for the calculation of metrics for the modelling

    Inputs:
    ----------
    data : pd.DataFrame
        Dataframe with data for the analysis
    active_tickers : list
        List of active tickers
    ma : float
        Critical value for MA
    mv : float
        Critical value for MV
    window : int
        The window size for MA and MV
    distance : int
        Minimal distance between critical transitions
    tail : int
        Tail of critical transitions that will separate it from noise outliers
    window_metrics : int
        The window size for metrics

    Returns:
    ----------
    ds : pd.DataFrame
        Dataframe with calculated values
    """

    # Calculating metrics
    ds = pd.DataFrame()
    for ticker in tqdm(active_tickers):
        data_ticker = data[[ticker, f'{ticker}, MA{window}', f'{ticker}, MV{window}', f'{ticker}, Rise']].dropna().reset_index(drop = True)
        data_ticker.rename(columns = {ticker: 'Volume', f'{ticker}, MA{window}': f'MA{window}', f'{ticker}, MV{window}': f'MV{window}', f'{ticker}, Rise': 'Rise'}, inplace = True)
        
        # Get indices of critical transitions
        rises_ticker = find_transitions(data_ticker, ma, mv, window, distance, tail).index

        # Iterating over tickers
        for ind in rises_ticker:
            ds_ticker_ind = data_ticker.iloc[ind - distance: ind + tail]
            ds_ticker_ind['Distance'] = - ds_ticker_ind.index + ind
            ds_ticker_ind['Index'] = ind
            ds_ticker_ind['Ticker'] = ticker
            ds_ticker_ind.reset_index(drop = True, inplace = True)

            Hurst = []
            corr_dim = []
            l_exp = []
            var = []
            skew = []
            kurt = []
            PSD = []
            acf_1 = []
            wl_c1 = []
            wl_c2 = []
            wl_c3 = []

            for j in range(len(ds_ticker_ind)):
                # Skipping first {window_metrics iterations which number is smaller than the needed window size
                if j <= window_metrics:
                    Hurst.append(None)
                    corr_dim.append(None)
                    l_exp.append(None)
                    var.append(None)
                    skew.append(None)
                    kurt.append(None)
                    PSD.append(None)
                    acf_1.append(None)
                    wl_c1.append(None)
                    wl_c2.append(None)
                    wl_c3.append(None)
                # Calculating metrics for the rest of the time series
                else:
                    data_before_j = ds_ticker_ind.iloc[j - window_metrics + 1 : j + 1]

                    Hurst.append(nolds.hurst_rs(data_before_j['Volume']))

                    corr_dim.append(nolds.corr_dim(data_before_j['Volume'], 10))

                    # Lyapunov exponent may throw an error if the series is too close to constant
                    # Lyapunov exponent may throw a ceil error if Nolds > 0.5.2 as l_exp is broken in 0.6.0 and 0.6.1
                    try:
                        l_exp.append(nolds.lyap_r(data_before_j['Volume']))
                    except:
                        l_exp.append(None)

                    var.append(data_before_j['Volume'].var())

                    skew.append(data_before_j['Volume'].skew())

                    kurt.append(data_before_j['Volume'].kurt())

                    freq_j, psd_j = sp.signal.welch(data_before_j['Volume'])
                    PSD.append(np.polyfit(np.log(freq_j)[1:j], np.log(psd_j)[1:j], 1)[0])

                    acf_1.append(sm.tsa.acf(data_before_j['Volume'], nlags = 1)[1])
                    
                    # Set of the available scaling ranges heavily depends on the size of the dataset
                    # So, we are checking better scaling range and then downsizing if it raises error
                    try:
                        dwt, lwt = mfa.mf_analysis_full(data_before_j['Volume'].values,
                            scaling_ranges = [(2, 5)],
                            q = build_q_log(1, 10, 20),
                            n_cumul = 3,
                            p_exp = np.inf,
                            gamint = 0.0
                        )
                    except:
                        dwt, lwt = mfa.mf_analysis_full(data_before_j['Volume'].values,
                            scaling_ranges = [(2, 4)],
                            q = build_q_log(1, 10, 20),
                            n_cumul = 3,
                            p_exp = np.inf,
                            gamint = 0.0
                        )
                    _, lwt_cumul, _, _ = lwt
                    wl_c1.append(lwt_cumul.log_cumulants[0][0][0])
                    wl_c2.append(lwt_cumul.log_cumulants[1][0][0])
                    wl_c3.append(lwt_cumul.log_cumulants[2][0][0])

            ds_ticker_ind['Hurst'] = Hurst
            ds_ticker_ind['CorrDim'] = corr_dim
            ds_ticker_ind['Lyapunov'] = l_exp
            ds_ticker_ind['Variance'] = var
            ds_ticker_ind['Skewness'] = skew
            ds_ticker_ind['Kurtosis'] = kurt
            ds_ticker_ind['PSD'] = PSD
            ds_ticker_ind['ACF_1'] = acf_1
            ds_ticker_ind['WL_C1'] = wl_c1
            ds_ticker_ind['WL_C2'] = wl_c2
            ds_ticker_ind['WL_C3'] = wl_c3
            # ds_ticker_ind.dropna(inplace = True)

            ds = pd.concat([ds, ds_ticker_ind])

    # Return calculated metrics
    ds.reset_index(drop = True, inplace = True)
    return ds

#---------------------------------------------------------------------------------------------------------------------------------------

def generate_features(data:pd.DataFrame,
                      cols:list,
                      lag_model:list) -> pd.DataFrame:
    """
    Function for the generation of features for the modelling

    Inputs:
    ----------
    data : pd.DataFrame
        Dataframe with data for the analysis
    cols : list
        List of columns for the analysis
    lag_model : list
        List of lags for the analysis

    Returns:
    ----------
    data_logdyn : pd.DataFrame
        Dataframe with calculated values
    """

    # Get unique indices
    indices = data.groupby(['Ticker', 'Index']).size().index.values

    # Calculate dynamics and short variance
    # Original idea about variance was born from the largest Lyapunov exponent's behaviour before the critical transition point:
    # is mostly didn't move in nominal values but its variance in some cases decreased signigicantly 
    data_logdyn = pd.DataFrame()
    for ind in tqdm(indices):
        data_ind = data[(data['Ticker'] == ind[0]) & (data['Index'] == ind[1])]
        for col in cols:
            for lag_m in lag_model:
                data_ind[col + '_' + str(lag_m) + '_dyn'] = data_ind[col] / data_ind[col].shift(lag_m) - 1
                data_ind[col + '_' + str(lag_m) + '_Variance'] = data_ind[col].rolling(lag_m).var()
        data_ind.dropna(inplace = True)
        data_logdyn = pd.concat([data_logdyn, data_ind])

    # Reset index to get rid of dates
    data_logdyn.reset_index(drop = True, inplace = True)
    data_logdyn = data_logdyn[data_logdyn['Distance'] > 0]

    return data_logdyn