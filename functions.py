import copy
import operator
import numpy as np
import scipy as sp
import pandas as pd
from tqdm import tqdm
import networkx as nx
import statsmodels.api as sm
import sklearn.metrics as metrics
import plotly.graph_objects as go
import sklearn.model_selection as modsel
from plotly.subplots import make_subplots

np.random.seed(0)

#---------------------------------------------------------------------------------------

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
                       silent:bool = False,
                       insignificant_feature:bool = True):
    """
    Function for the optimization of OLS

    Inputs:
    ----------
    Y_train, Y_test : array
        Target variable for the regression
    X_train, X_test : DataFrame
        Set of X for the model
    type : str = 'Probit'
        Type of the model
    p_value_bord : float = 0.05
        Maximum acceptable p-value for the coefficient
    silent : bool = False
        Whether not to show reports about model
    insignificant_feature : bool = True
        Whether to drop insignificant features or to keep them

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
    
    if insignificant_feature == False:
        # Create model
        if type == 'Probit':
            model = sm.Probit(Y_train, X_train)
        else:
            model = sm.Logit(Y_train, X_train)
        results = model.fit(disp = 0)

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
                       groupby:str,
                       mean_only:bool = False):

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
        fig.add_trace(go.Scatter(x = data_mean.index, y = data_median[col], mode = 'lines', name = f'{col}_median', line = dict(color = 'yellow')), row = i + 1, col = 1)
        if mean_only != True:
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

#---------------------------------------------------------------------------------------
    
def model(data,
          target:str,
          horizons:list,
          shares:list,
          states:list,
          separate:bool = False):
    
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
    separate : bool = False
        Whether to calculate whole models or to separate variables to different models

    Returns:
    --------------------
    res : pd.DataFrame
        Dataframe with raw statistical results of the modelling
    """

    columns = ['Horizon', '1 Share', '1 Share real', 'State',
               'Train size', 'Test size', 'Train AUC', 'Test AUC',
               'Train KS-test p-value', 'Test KS-test p-value',
               'Train F1-score', 'Test F1-score', 
               'Train precision', 'Test precision', 
               'Train recall', 'Test recall', 'Coeffs']
    if separate == True:
        columns = columns[:4] + ['Variable'] + columns[4:] + ['Pvalues']

    # Create dataframe for the results
    res = pd.DataFrame(columns = columns)

    # Iterate over the chosen parameters and optimize classification models, then save all the results to the dataframe
    for horizon in tqdm(horizons):
        data_testing = data.copy()
        data_testing['Flag'] = data_testing['Distance'].apply(lambda x: 0 if x > horizon else 1)
        data_testing.drop(columns = ['Volume', 'MA100', 'Rise', 'Distance', 'Index', 'Ticker'], inplace = True)
        
        data_testing_1 = data_testing[data_testing[target] == 1]
        data_testing_0 = data_testing[data_testing[target] == 0]
        Y_1 = data_testing_1[target]
        if separate == False:
            X_1 = data_testing_1.drop(columns = [target])
            share_1_orig = len(data_testing_1) / (len(data_testing_0) + len(data_testing_1))
            for share in shares:
                for state in states:
                    _, X_0, _, Y_0 = modsel.train_test_split(data_testing_0.drop(columns = [target]), data_testing_0[target], 
                                                             test_size = min(share_1_orig * (1 - share) / share, 1), random_state = state)
                    share_1 = len(Y_1) / (len(Y_0) + len(Y_1))
                    Y = pd.concat([Y_0, Y_1])
                    X = sm.add_constant(pd.concat([X_0, X_1]))
                    X_train, X_test, Y_train, Y_test = modsel.train_test_split(X, Y, test_size = 0.2, random_state = state)
                    results_rs, auc_train_rs, auc_test_rs, ks_train_rs, ks_test_rs, f1_train_rs,\
                        f1_test_rs, pr_train_rs, pr_test_rs, rec_train_rs, rec_test_rs\
                        = model_optimization(Y_train, Y_test, X_train, X_test, silent = True)
                    res.loc[len(res)] = [horizon, share, share_1, state, len(Y_train), len(Y_test),
                                        auc_train_rs, auc_test_rs, ks_train_rs, ks_test_rs,
                                        f1_train_rs, f1_test_rs, pr_train_rs, pr_test_rs,
                                        rec_train_rs, rec_test_rs, results_rs.params]
        else:
            for col in data_testing.columns.drop(target):
                X_1 = data_testing_1[col]
                share_1_orig = len(data_testing_1) / (len(data_testing_0) + len(data_testing_1))
                for share in shares:
                    for state in states:
                        _, X_0, _, Y_0 = modsel.train_test_split(data_testing_0[col], data_testing_0[target], 
                                                                 test_size = min(share_1_orig * (1 - share) / share, 1), random_state = state)
                        share_1 = len(Y_1) / (len(Y_0) + len(Y_1))
                        Y = pd.concat([Y_0, Y_1])
                        X = sm.add_constant(pd.concat([X_0, X_1]))
                        X_train, X_test, Y_train, Y_test = modsel.train_test_split(X, Y, test_size = 0.2, random_state = state)
                        try:
                            results_rs, auc_train_rs, auc_test_rs, ks_train_rs, ks_test_rs, f1_train_rs,\
                                f1_test_rs, pr_train_rs, pr_test_rs, rec_train_rs, rec_test_rs\
                                = model_optimization(Y_train, Y_test, X_train, X_test, silent = True,
                                                    insignificant_feature = False)
                            res.loc[len(res)] = [horizon, share, share_1, state, col, len(Y_train), len(Y_test),
                                                auc_train_rs, auc_test_rs, ks_train_rs, ks_test_rs,
                                                f1_train_rs, f1_test_rs, pr_train_rs, pr_test_rs,
                                                rec_train_rs, rec_test_rs, results_rs.params, results_rs.pvalues]
                        except:
                            pass
            
    return res

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
                 type:str = 'BTW'):
    
    """
    Function for the implementation of different spread models on the graph.
    
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
    type : str
        The type of model to be used. Default is 'BTW'. Other option: 'MA'.
    
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

    return G, falls_d

#---------------------------------------------------------------------------------------

def spread(model:str,
           G:nx.Graph,
           number_of_days:int,
           new_grains:list,
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
    ad_dissipation : bool
        If True, the spread model includes ad dissipation (+2-1 instead of +1-0).
    neutral_state : bool
        If True, the spread model includes neutral state (+1-1 instead of +1-0).
    new_grains_plus : list
        A list of lists, where each inner list contains the nodes where new grains are added on each day.
    new_grains_minus : list
        A list of lists, where each inner list contains the nodes where new grains are subtracted on each day.

    Returns:
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
    for j in range(0, number_of_days):
        status['day'+str(j)] = np.zeros(G.number_of_nodes())

    # Set node attributes
    node_attr = status.to_dict('index')
    nx.set_node_attributes(G, node_attr)

    # Simulate the spread
    for d in tqdm(range(0, number_of_days-1), disable = silent):
        falls_d = 0

        # Add/subtract grains
        if ad_dissipation == True:
            for node in G.nodes(data=True):
                if node[0] in new_grains[d]:
                    node[1]['day'+str(d)] += 1.0
                if node[0] in new_grains_plus[d]:
                    node[1]['day'+str(d)] += 1.0
                if node[0] in new_grains_minus[d]:
                    node[1]['day'+str(d)] -= 1.0          
                node[1]['day'+str(d+1)] += node[1]['day'+str(d)]
        elif neutral_state == True:
            for node in G.nodes(data=True):
                if node[0] in new_grains[d]:
                    node[1]['day'+str(d)] += 1.0
                if node[0] in new_grains_minus[d]:
                    node[1]['day'+str(d)] -= 1.0          
                node[1]['day'+str(d+1)] += node[1]['day'+str(d)]    
        else:
            for node in G.nodes(data=True):
                if node[0] in new_grains[d]:
                    node[1]['day'+str(d)] += 1.0
                node[1]['day'+str(d+1)] += node[1]['day'+str(d)]

        # Spread grains
        for node in G.nodes(data=True):
            if d <= (number_of_days-1) and node[1]['day'+str(d)] >= deg[node[0]] and deg[node[0]] > 0:
                G, falls_d = spread_model(G, ones, falls_d, d, node, deg[node[0]], type = model)

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