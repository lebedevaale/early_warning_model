import operator
import numpy as np
import scipy as sp
import pandas as pd
import statsmodels.api as sm
import sklearn.metrics as metrics
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def roc_metric(Y, 
               Y_pred):
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
                              results):
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
    max_p_value = max(results.pvalues.iteritems(), key = operator.itemgetter(1))[0]
    # Drop the worst feature
    X.drop(columns = max_p_value, inplace = True)
    X_test.drop(columns = max_p_value, inplace = True)

    return X, X_test

#---------------------------------------------------------------------------------------

def model_optimization(Y_train,
                       Y_test,
                       X_train,
                       X_test,
                       type:str = 'Probit', 
                       p_value_bord:float = 0.05, 
                       silent:bool = False):
    """
    Function for the optimization of OLS

    Inputs:
    ----------
    Y : array
        Target variable for the regression
    X : DataFrame
        Set of X for the model
    type : str = 'Probit'
        Type of the model
    p_value_bord : float = 0.05
        Maximum acceptable p-value for the coefficient
    silent : bool = False
        Whether not to show reports about model

    Returns:
    ----------
    results : model
        Fitted statsmodels model
    auc_train : float
        AUC on the train data
    auc_test : float
        AUC on the test data
    ks_train.pvalue : float
        KS-test p-value on the train data
    ks_test.pvalue : float
        KS-test p-value on the test data
    f1_train : float
        F1-score on the train data
    f1_test : float
        F1-score on the test data
    pr_train : float
        Precision score on the train data
    pr_test : float
        Precision score on the test data
    rec_train : float
        Recall score on the train data
    rec_test : float
        Recall score on the test data
    """
    
    insignificant_feature = True
    while insignificant_feature:
        # Create model
        if type == 'Probit':
            model = sm.Probit(Y_train, X_train)
        else:
            model = sm.Logit(Y_train, X_train)

        # Fit model and get
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
    
    Y_train_pred = results.predict(X_train)
    Y_test_pred = results.predict(X_test)
    auc_train, threshold_train = roc_metric(Y_train, Y_train_pred)
    auc_test, threshold_test = roc_metric(Y_test, Y_test_pred)
    Y_train_pred_round = np.where(Y_train_pred < threshold_train, np.floor(Y_train_pred), np.ceil(Y_train_pred))
    Y_test_pred_round = np.where(Y_test_pred < threshold_test, np.floor(Y_test_pred), np.ceil(Y_test_pred))

    ks_samples_train = pd.DataFrame({'Y': Y_train, 'Y_pred': Y_train_pred})
    ks_samples_train_posi = ks_samples_train[ks_samples_train['Y'] == 1]['Y_pred']
    ks_samples_train_nega = ks_samples_train[ks_samples_train['Y'] == 0]['Y_pred']
    ks_train = sp.stats.kstest(ks_samples_train_posi, ks_samples_train_nega)
    ks_samples_test = pd.DataFrame({'Y': Y_test, 'Y_pred': Y_test_pred})
    ks_samples_test_posi = ks_samples_test[ks_samples_test['Y'] == 1]['Y_pred']
    ks_samples_test_nega = ks_samples_test[ks_samples_test['Y'] == 0]['Y_pred']
    ks_test = sp.stats.kstest(ks_samples_test_posi, ks_samples_test_nega)

    f1_train = round(metrics.f1_score(Y_train, Y_train_pred_round), 3)
    f1_test = round(metrics.f1_score(Y_test, Y_test_pred_round), 3)
    pr_train = round(metrics.precision_score(Y_train, Y_train_pred_round), 3)
    pr_test = round(metrics.precision_score(Y_test, Y_test_pred_round), 3)
    rec_train = round(metrics.recall_score(Y_train, Y_train_pred_round), 3)
    rec_test = round(metrics.recall_score(Y_test, Y_test_pred_round), 3)
    if silent == False:
        print(f'''Train AUC score: {auc_train}, Train KS-test p-value: {round(ks_train.pvalue, 3)}, 
              Train F1-score: {f1_train}, Train precision: {pr_train}, Train recall: {rec_train}''')
        print(f'''Test AUC score: {auc_test}, Test KS-test p-value: {round(ks_test.pvalue, 3)}, 
              Test F1-score: {f1_test}, Test precision: {pr_test}, Test recall: {rec_test}''')
        print(results.summary())

    return results, auc_train, auc_test, round(ks_train.pvalue, 9), round(ks_test.pvalue, 9),\
           f1_train, f1_test, pr_train, pr_test, rec_train, rec_test

#---------------------------------------------------------------------------------------

def heatmap(data):

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
        template = 'plotly_dark',
        font = dict(size = 14),
        height = 600,
        width = 1600
    )
    fig.update_annotations(font_size = 30)

    # Show the plot
    fig.show()

#---------------------------------------------------------------------------------------

def variables_dynamics(data,
                       groupby):

    """
    Function for the plotting of the dynamics for the variables

    Inputs:
    --------------------
    data : pd.DataFrame
        Dataframe with columns for the analysis
    groupby : str
        Column to groupby

    Prints:
    --------------------
    Dynamics of the variables
    """

     # Creating grid of subplots
    av_cols = data.drop(columns = [groupby]).columns
    fig = make_subplots(rows = len(av_cols), cols = 1, subplot_titles = [col for col in av_cols])

    # Calculating mean, min and max values of variables for each unique value in groupby column
    data_mean = data.groupby(groupby).mean()
    data_min = data.groupby(groupby).min()
    data_max = data.groupby(groupby).max()

    # Scattering returns
    for i, col in enumerate(av_cols):
        fig.add_trace(go.Scatter(x = data_mean.index, y = data_mean[col], mode = 'lines', name = f'{col}_mean', line = dict(color = 'green')), row = i + 1, col = 1)
        fig.add_trace(go.Scatter(x = data_min.index, y = data_min[col], mode = 'lines', name = f'{col}_min', line = dict(color = 'red')), row = i + 1, col = 1)
        fig.add_trace(go.Scatter(x = data_max.index, y = data_max[col], mode = 'lines', name = f'{col}_max', line = dict(color = 'blue')), row = i + 1, col = 1)
        fig.update_xaxes(autorange = "reversed", row = i + 1, col = 1)

    # Update layout
    fig.update_layout(
        showlegend = False,
        template = 'plotly_dark',
        font = dict(size = 20),
        height = 300 * len(av_cols),
        width = 1200
    )

    # Show the plot
    fig.show()