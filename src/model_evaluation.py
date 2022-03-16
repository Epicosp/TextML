# imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_text
from tensorflow import keras
import seaborn as sns


def confusion_matrix(model, x_test, y_test, model_name):
    '''
    generates predictions from keras.model.predict and creates a confusion matrix of size equal to the amount of possible predictions.

    Returns a heatmap relating to the confusion matrix. The matrix values will be adjusted by the weight of each class in the test data.

    '''
    #generate predictions using test data
    predictions = model.predict(x_test)

    # ensure the data is in the correct structure
    predictions = pd.DataFrame(predictions)
    results = pd.DataFrame()

    # find column of max for each row and make new column for 'P' Prediction
    results['P'] = predictions.idxmax(axis = 1)

    # reset index of y_test and append actual encoded values as 'A', 
    y_test = y_test.reset_index(drop=True)
    results['A'] = y_test

    # calculate a frequency value for each P/A pair.
    results = results.groupby(results.columns.tolist(),as_index=False).size()

    size = len(predictions.columns.tolist())
    results_array = results.to_numpy()

    #generate confusion matrix grid (n x n)
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

    # # calculate matrix adjustements by observing weight of each class in the test data
    # counts = y_test.groupby('y_test').size()
    # counts.rename(columns = {'size':'y_test_count'}, inplace=True)
    # total_count = len(y_test)
    # counts["adjustment_factor"] = counts['y_test_count']/total_count

    # # apply the adjustment factor along rows
    # mtx = mtx.multiply(counts["adjustment_factor"], axis = 'index')
        
    sns.heatmap(mtx, annot=True)
    plt.savefig(f'{model_name}/confusion_matrix.png', dpi = 400)


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
