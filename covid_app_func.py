import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import seaborn as sns
from scipy.stats import boxcox
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.graphics.gofplots import qqplot
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

@st.cache
def process_data(path, column):
    # load data
    df = pd.read_csv(path, index_col=None)
    df = df.loc[32:,['date', 'days_elapse', 'new_count_update']]
    df.reset_index(inplace=True, drop=True)
    df.columns = ['date', 'days_elapse', 'daily_case']

    # add a 14-day rolling mean and std dev
    df['daily_case_mean'] = df['daily_case'].rolling(window = 14).mean()
    df['daily_case_stdev'] = df['daily_case'].rolling(window = 14).std()

    return df

# add n-day mean for modelling
def mean_data(df_in, n):
    n_mod = str(n) + 'D'
    df_out = df_in.copy()
    df_out.set_index('date', inplace=True, drop = True)
    df_out.index = pd.to_datetime(df_out.index)
    df_out = df_out.resample(n_mod).mean()
    df_out['days_elapse'] = df_out['days_elapse'] + (n-1)/2
    df_out.reset_index(inplace=True, drop = True)

    # plot
    n_label = str(n) + '-Day Mean'
    fig_out = plt.figure()
    plt.plot(df_in['days_elapse'], df_in['daily_case'], color = 'blue', label = 'Daily')
    plt.plot(df_out['days_elapse'], df_out['daily_case'], color = 'red', label = n_label)
    plt.legend(loc = 'best')
    plt.title(n_label)
    
    return df_out, fig_out

# define a function to add various columns transformations to the dataframe
# includes: differencing, boxcox, differencing of boxcox, 2 week rolling normalization and differencing
@st.cache
def add_series(df, columns_analyze):
    for column in df:
        if column in columns_analyze:
           # create a column for differences
            df[column + "_diff"] = df[column] - df[column].shift(1)
            # create a column with a box-cox transform
            df[column + "_bc"], lam = boxcox(df[column])
            # create a column with differences of box-cox transform (add 200 before applying the transform to avoid neg values)
            df[column + "_bc_diff"] = df[column + "_bc"] - df[column + "_bc"].shift(1)
            # create a 2 week rolling normalized column
            df[column + "_2wknormal"] = (df[column] - df[column + '_mean']) / df[column + '_stdev']
            # create a column for difference of 2 week normalized column
            df[column + "_2wknormal_diff"] = df[column + "_2wknormal"] - df[column + "_2wknormal"].shift(1)
            
            #print('Lambda of Box-Cox Transform: %f' % lam)
            
    return df, lam

# function to reverse a box-cox transform
def boxcox_inverse(value, lam):
    if (lam * value + 1) <= 0:
        return 0
    if lam == 0:
        return math.exp(value)
    return math.exp(math.log(lam * value + 1) / lam)

# function to view plots of data
def explore_series(df, columns_explore, plot_lags=20):
    # Figure setup
    fig_out = plt.figure(figsize = (22, 24))
    plot_cols = len(columns_explore)
    position = 0
    # create lists for output values from adfuller tests
    output_ADF = []
    output_pval = []
    output_crit1 = []
    output_crit5 = []
    output_crit10 = []
    output_labels = []
    
    # loop through the columns of interest
    for column in df:
        if column in columns_explore:
            # time history plot
            df[column].plot(ax = plt.subplot2grid((5, plot_cols), (0, position)), title = column)
            # histogram
            df[column].hist(ax = plt.subplot2grid((5, plot_cols), (1, position)))
            # qqplot to check normality
            qqplot(df[column], line='r', ax = plt.subplot2grid((5, plot_cols), (2, position)))
            # autocorrelation plot
            plot_acf(df[column].dropna(), lags=plot_lags, ax=plt.subplot2grid((5, plot_cols), (3, position)))
            # partial autocorrelation plot
            plot_pacf(df[column].dropna(), lags=plot_lags, ax=plt.subplot2grid((5, plot_cols), (4, position)))
            position += 1
            
            # run adfuller test and append results to lists
            result = adfuller(df[column].dropna())
            output_ADF.append(result[0])
            output_pval.append(result[1])
            output_crit1.append(result[4]['1%'])
            output_crit5.append(result[4]['5%'])
            output_crit10.append(result[4]['10%'])
            output_labels.append(column)
     
    # create dataframe for the adfuller results
    df_out = pd.DataFrame(columns=output_labels, index=['ADF_Statistic', 'p-value', 'Critical_1percent', 
                                                 'Critical_5_percent', 'Critical_10_percent'])
    df_out.iloc[0] = output_ADF
    df_out.iloc[1] = output_pval
    df_out.iloc[2] = output_crit1
    df_out.iloc[3] = output_crit5
    df_out.iloc[4] = output_crit10  
            
    return df_out, fig_out

# calculate repeated train-test splits of time series data
def splitter(df, column, split_num):
    splits = TimeSeriesSplit(n_splits=split_num)
    fig_out = plt.figure(figsize = (16, 5*split_num))
    index = 1
    for train_index, test_index in splits.split(df[column]):
        train = df[column][train_index]
        test = df[column][test_index]
        #print('Split %d Observations: %d' % (index, (len(train) + len(test))))
        #print('Training Observations: %d' % (len(train)))
        #print('Testing Observations: %d' % (len(test)))
        plt.subplot(810 + index*2)
        plt.plot(train)
        plt.plot([None for i in train] + [x for x in test])
        index += 1
    plt.show()
    return fig_out

# null model" persistance model
def persistence(train, test, label=""):
    name = 'pers_predict_' + label
    combine = pd.concat([train, test]).to_frame()
    combine.columns = ['test']
    combine[name] = combine['test'].shift()
    combine = combine.iloc[test.index]
    rmse = math.sqrt(mean_squared_error(combine[name], combine['test']))
    #print('RMSE: %.3f' % rmse)
    #combine.plot()
    #plt.show()
    return combine[[name]], rmse

# null model: rolling mean model
def roller(train, test, window_len=3, label=""):
    name = 'roll_predict_' + label
    combine = pd.concat([train, test]).to_frame()
    combine.columns = ['test']
    combine[name] = combine['test'].shift().rolling(window_len).mean()
    combine = combine.iloc[test.index]
    rmse = math.sqrt(mean_squared_error(combine[name], combine['test']))
    #print('RMSE: %.3f' % rmse)
    #combine.plot()
    #plt.show()
    return combine[[name]], rmse

# ARIMA model for a given order (p,d,q) 
# with number of steps to predict and optional confidence intervals
# uses a boxcox transform
def arima_model(data, arima_order, predict_steps = 1, confidence = False):
    # transform
    transformed, lam = boxcox(data)
    if lam < -5:
        transformed, lam = data, 1
    # predict    
    model = ARIMA(transformed, order=arima_order)
    model_fit = model.fit(disp=0)
    temp_results = model_fit.forecast(steps = predict_steps)
    yhat = temp_results[0]
    if confidence == True:
        ylow = temp_results[2].transpose()[0]
        yhigh = temp_results[2].transpose()[1]
    # invert transformed prediction
    predictions = [boxcox_inverse(i, lam) for i in yhat]
    if confidence == True:
        low = [boxcox_inverse(i, lam) for i in ylow]
        high = [boxcox_inverse(i, lam) for i in yhigh]
        return predictions, low, high
    return predictions

# evaluate an ARIMA model using a rolling forward method
# with number of steps to predict and optional confidence intervals
def evaluate_arima_model(train, test, arima_order, label="", predict_steps = 1, confidence = False):
    name = []
    step_name = [('arima_predict_' + label + "_step" + str(i)) for i in np.arange(1,(predict_steps+1))]
    name.append(step_name)
    ### the label below is for a different type of prediction currently not in use
#   stepb_name = [('arima_predict_' + label + "_step" + str(i) + "b") for i in np.arange(2,(predict_steps+1))]
#   name.append(stepb_name)
    name = [val for sublist in name for val in sublist]
            
    history = [x for x in train]
    # make predictions
    predictions = []
    if confidence == True:
        low = []
        high = []
    for t in range(len(test)):
        if confidence == True:
            yhat, ylow, yhigh = arima_model(history, arima_order, predict_steps, confidence)
        else:
            yhat = arima_model(history, arima_order, predict_steps, confidence)

        ### uses the output from the previous model to run the next model
        ### currently not being used
#        for i in np.arange(1,predict_steps):
#            history.append(yhat[i-1])
#            temp_output = arima_model(history, arima_order, predict_steps = 1)
#            yhat.insert(i, temp_output[0])
            
#        if predict_steps > 1:
#            del history[-(predict_steps - 1):]
            
        predictions.append(yhat)
        if confidence == True:
            low.append(ylow)
            high.append(yhigh)
        # observation
        obs = test[t+test.index.min()]
        history.append(obs)
        #print('>Predicted=%.3f, Expected=%.3f' % (yhat, obs))
    # calculate out of sample error for the first step
    predictions = pd.DataFrame(predictions, index = test.index, columns = name)
    rmse = math.sqrt(mean_squared_error(test, predictions.iloc[:,0]))
    if confidence == True:
        low = pd.DataFrame(low, index = test.index, columns = name)
        high = pd.DataFrame(high, index = test.index, columns = name)
        return predictions, rmse, low, high
    return predictions, rmse

# grid search ARIMA parameters for time series
# evaluate combinations of p, d and q values for an ARIMA model
# only uses a single step prediction
def grid_ARIMA_models(train, test, p_values, d_values, q_values):
    best_score, best_cfg = float("inf"), None
    p_list = []
    d_list = []
    q_list = []
    rmse_list = []
    for p in p_values:
        for d in d_values:
            for q in q_values:
                order = (p,d,q)
                p_list.append(p)
                d_list.append(d)
                q_list.append(q)
                try:
                    _, rmse = evaluate_arima_model(train, test, order)
                    rmse_list.append(rmse) 
                    if rmse < best_score:
                        best_score, best_cfg = rmse, order
                    #print('ARIMA%s RMSE=%.3f' % (order,rmse))
                except:
                    rmse_list.append(np.NaN) 
                    continue
    best = ('Best ARIMA%s RMSE=%.3f' % (best_cfg, best_score))
    output = pd.DataFrame(list(zip(p_list, d_list, q_list, rmse_list)), columns = ['p', 'd', 'q', 'rmse'])
    return output, best

# heat map of the model performance for the ARIMA grid search
def pdq_heat(df):
    position = 0
    d_list = df['d'].unique()
    fig_out = plt.figure(figsize = (22, 8))
    plot_cols = len(d_list)
    for d in d_list:
        temp_df = df[df['d']==d].drop(columns='d')
        temp_df = temp_df.pivot_table(columns = 'p', index = 'q', values = 'rmse')
        sns.heatmap(temp_df, annot=True, ax = plt.subplot2grid((1, plot_cols), (0, position)))
        position += 1
    return fig_out

# evaluate the null models and the ARIMA model
# only for single step ARIMA evaluation
def evaluate_models(splits, df, data_col, arima_order):
    idx = 0
    out_data = df.copy()
    pers_rmse = []
    roll_rmse = []
    arima_rmse = []
    col_labels = []
    for train_index, test_index in splits.split(df[data_col]):
        idx += 1
        col_labels.append('split_' + str(idx))
        
        train = df[data_col][train_index]
        test = df[data_col][test_index]
        
        temp_results, temp_rmse = persistence(train, test, label = str(idx))
        out_data = pd.merge(out_data, temp_results, left_index = True, right_index = True, how = 'outer')
        pers_rmse.append(temp_rmse)
        
        temp_results, temp_rmse = roller(train, test, label = str(idx))
        out_data = pd.merge(out_data, temp_results, left_index = True, right_index = True, how = 'outer')
        roll_rmse.append(temp_rmse)
        try:
            temp_results, temp_rmse = evaluate_arima_model(train, test, arima_order, label = str(idx), predict_steps = 1)
            out_data = pd.merge(out_data, temp_results, left_index = True, right_index = True, how = 'outer')
            arima_rmse.append(temp_rmse)
        except:
            arima_rmse.append(np.NaN) 
            continue

    out_rmse = pd.DataFrame(columns=col_labels, index=['Persistence', 'Roller', 'ARIMA'])    
    out_rmse.iloc[0] = pers_rmse
    out_rmse.iloc[1] = roll_rmse
    out_rmse.iloc[2] = arima_rmse     
        
    return out_data, out_rmse

# plot to compare results from different prediction models
def compare_plot(df, index_col, labels, original):
    df_mod = df.set_index(index_col, drop = True)
    plot_rows = len(labels)
    fig_out = plt.figure(figsize = (22, 4*plot_rows))
    position = 0 
    for label in labels:
        look_for = label + "|" + original
        include = df_mod.columns.str.contains(look_for, case=False)
        temp_df = df_mod.iloc[:, include]
        temp_df.plot(ax = plt.subplot2grid((plot_rows, 1), (position, 0)))
        position += 1
    return fig_out

# model multistep ARIMA forecast
# optional confidence interval which creates two additional output dfs
def multistep_forecast(splits, df, data_col, arima_order, forecast_steps, confidence = False):
    idx = 0
    out_data = df.copy()
    # create dataframes for the low and high confidence intervals if required
    if confidence == True:
        out_low_data = df.copy()
        out_high_data = df.copy()
    out_rmse = []
    col_labels = []
    index_labels = [("step" + str(i)) for i in np.arange(1,(forecast_steps+1))]
    for train_index, test_index in splits.split(df[data_col]):
        idx += 1
        col_labels.append('split_' + str(idx))

        train = df[data_col][train_index]
        test = df[data_col][test_index]
    
        if confidence == False:
            try:    
                temp_results, _ = evaluate_arima_model(train, test, arima_order, label = str(idx), 
                                                        predict_steps = forecast_steps, confidence = False)
            except:
                continue
        # include low and high confidence limits if required
        else:
            try:
                temp_results, _, low_temp, high_temp = evaluate_arima_model(train, test, arima_order, label = str(idx), 
                                                        predict_steps = forecast_steps, confidence = True)
                out_low_data = pd.merge(out_low_data, low_temp, left_index = True, right_index = True, how = 'outer')
                out_high_data = pd.merge(out_high_data, high_temp, left_index = True, right_index = True, how = 'outer')
            except:
                continue
            
        out_data = pd.merge(out_data, temp_results, left_index = True, right_index = True, how = 'outer')
    
        # shift the output step results to have the correct index
        # "start" is the column location of the first step
        start = 2 + (forecast_steps * (idx - 1))
        # create a temp dataframe to compare the rmse with the original data
        temp_rmse = []
        for i in np.arange(forecast_steps):
            out_data.iloc[:,start+i] = out_data.iloc[:,start+i].shift(i)
            # shift confidence limit dataframes if required
            if confidence == True:
                out_low_data.iloc[:,start+i] = out_low_data.iloc[:,start+i].shift(i)
                out_high_data.iloc[:,start+i] = out_high_data.iloc[:,start+i].shift(i)
            # create the temp df for rmse comparison and append calculated rmse    
            temp_df = pd.concat([out_data.iloc[:, start+i], out_data.loc[:,'daily_case']], axis = 1)
            temp_df.dropna(inplace = True)
            rmse = math.sqrt(mean_squared_error(temp_df.iloc[:, 0], temp_df.iloc[:, 1]))
            temp_rmse.append(rmse)
    
        out_rmse.append(temp_rmse)
    
    out_rmse = pd.DataFrame(out_rmse).transpose()
    out_rmse.columns = col_labels
    out_rmse.set_index(pd.Index(index_labels), inplace = True)
    
    if confidence == False:
        return out_data, out_rmse
    else:
        return out_data, out_rmse, out_low_data, out_high_data


