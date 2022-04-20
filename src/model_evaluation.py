# imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_text
from tensorflow import keras
import seaborn as sns

def results(predictions, y_test):
    '''Returns a dataframe containing predicted and true values from keras.model.predict object.'''

    # open empty dataframes, ensure data is in correct form
    predictions = pd.DataFrame(predictions)
    results = pd.DataFrame()

    # find column of max for each row and make new column for 'P' Prediction
    results['P'] = predictions.idxmax(axis = 1)

    # reset index of y_test and append actual encoded values as 'A', 
    y_test = y_test.reset_index(drop=True)
    results['A'] = y_test

    return results


def confusion_matrix(results):
    '''Confusion matrix of size equal to the amount of possible predictions. Returns a pandas dataframe.'''

    # calculate a frequency value for each P/A pair.
    results = results.groupby(results.columns.tolist(),as_index=False).size()

    # Convert to np array
    results_array = results.to_numpy()

    # Generate matrix grid (n x n)
    size = len(list(results['A'].unique()))
    mtx = {}
    for i in range(size):
        for x in range(size):    
            mtx[i] = {i:np.NaN}
    mtx = pd.DataFrame(mtx)

    # iterate over the np array, use the P,A values as coordinates to insert the respective size value
    # any missing values will remain as NaN. once filled, replace NaN with 0 for the heatmap.
    # matrix is represented with Predictions on the x axis (columns) and True values on the y axis (rows)
    for item in results_array:
        mtx[item[0]][item[1]] = item[2]
    mtx = mtx.replace(np.NaN, 0)
     
    return mtx 

def conf_mtx_weights (confusion_matrix, y_test):
    '''
    Applies adjustments to the values in a confusion matrix based
    
    on the relative frequency of classification types in the testing data

    type(confusion_matrix) pandas.DataFrame = Dataframe object containing neumerical values representing a confusion matrix.

    type(y_test) pandas.series.Series = set of testing data with unique objects equal to the classes in the confusion matrix.

    '''
    # New dataframe to allow application groupby function
    counts = pd.DataFrame()

    # Pass in data and apply groupby, store size values
    counts['y_test'] = y_test.reset_index(drop=True)
    counts = counts.groupby('y_test').size()

    # Operations for calculation of adjustment factor for each classification type
    total_count = len(y_test)
    adjustment_factor = counts/total_count

    # apply the adjustment factor along rows
    mtx = confusion_matrix.multiply(adjustment_factor, axis = 'index')
    
    return mtx

def compute_accuracy(results):
    '''compute keras.metrics.accuracy, return accuracy value'''
    acc = tf.keras.metrics.Accuracy()
    acc.update_state(results['P'], results['A'])
    return acc.result().numpy()

def save_model_data(model, model_evaluation, model_history, model_name):
    '''
    saves the evaluation, loss and acuracy into a csv file.

    saves the trained model (warning large file size)

    saves the model architecture summary as a .txt file
    '''
    # initilize df1 and add evaluation results
    eval = pd.DataFrame(model_evaluation)
    eval = eval.T
    eval['epoch'] = 'evaluation'
    eval = eval.rename(columns = {0:'loss',1:'accuracy'})

    # initilize df2 and add .fit metrics
    history = pd.DataFrame(model_history.history)
    history['epoch'] = np.arange(len(history))
    history['epoch'] = history['epoch'].apply(lambda x: x + 1)

    # combine df1 and df2
    data = history.append(eval)

    # export to csv
    data.to_csv(f'{model_name}/evaluation.csv')

    # save the model
    model.save(f'{model_name}/model')

    # save model architecture as .txt
    with open(f'{model_name}/modelsummary.txt', 'w') as f:
        model.summary(print_fn=lambda x: f.write(x + '\n'))

def compute_precision(confusion_matrix):
    '''
    extracts the diagonal from the confusion matrix and divides by the sum of the rows. 
    
    returns a list of precisions for each class in the data.
    '''
    precision = np.diag(confusion_matrix) / np.sum(confusion_matrix, axis = 0)
    return precision

def compute_recall(confusion_matrix):
    '''
    extracts the diagonal from the confusion matrix and divides by the sum of the columns. 
    
    returns a list of recalls for each class in the data.
    '''
    recall = np.diag(confusion_matrix) / np.sum(confusion_matrix, axis = 1)
    return recall