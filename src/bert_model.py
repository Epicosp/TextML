#imports
from typing import Type
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text
import pandas as pd
import numpy as np
import warnings
from time import time

class BertModel:
    def __init__(self, X_train, y_train, X_test, y_test, num_catagories, model_name='BertModel'):
        '''
        constructs a tf.keras neural network using bert model for text classification
        model_name (string): name used when generating save files
        X_train (pandas.core.series.Series): x axis training data
        y_train (pandas.core.series.Series): y axis training data
        X_test (pandas.core.series.Series): x axis testing data
        y_test (pandas.core.series.Series): y axis testing data
        num_catagories (int): integer value for the nuumber of training catagories
        ''' 
        # Checks
        if not all(isinstance(i, pd.core.series.Series) for i in [X_train, y_train, X_test, y_test]):
            raise TypeError("input data must be a pandas.core.series.Series ")
        if not isinstance(num_catagories, int):
            raise TypeError("num_catagories must be an interger describing the amount of training catagories.")
        if len(X_train) != len(y_train):
            raise Exception('training data sets are not the same size!')
        if len(X_test) != len(y_test):
            raise Exception('testing data sets are not the same size!')
        if len(X_test) > len(X_train):
            warnings.warn('Your testing dataset is larger than your training data set, is this intentional?')

        self.name = model_name
        self.X_test = X_test
        self.y_test = y_test
        self.X_train = X_train
        self.y_train = y_train
        self.num_catagories = num_catagories
        self.trained = False

    def generate_model(self):
        '''
        NLP architechture consisting of BERT (v4) preprocessing and encoder,
        an input layer with neurons equal to the BERT output length,
        A dropout layer and an output layer with neurons equal to num_catagories.
        
        returns a model ready for training data and prints model architecture to console
        '''
        # Check if model already exists, (eg from load_model or previous instance of generate_model)
        if hasattr(self, 'model'):
            raise Exception('model already exists for this object, cannot generate model')

        #BERT base (v4) preprocessing and encoder.
        preprocess_url = "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3"
        encoder_url = "https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/4"
        bert_preprocess_model = hub.KerasLayer(preprocess_url)
        bert_encoder = hub.KerasLayer(encoder_url)

        #placeholder for initial tensor
        text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text')

        #generate the bert encodings
        preprocessed_text = bert_preprocess_model(text_input)
        outputs = bert_encoder(preprocessed_text)

        #model architecture
        layer = tf.keras.layers.Dropout(0.1, name='dropout')(outputs['pooled_output'])
        layer = tf.keras.layers.Dense(self.num_catagories, activation='sigmoid', name='output')(layer)
        model = tf.keras.Model(inputs=[text_input], outputs = [layer])

        # compile
        model.compile(optimizer="adam", loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=["accuracy"])

        #set attribute to model object
        self.model = model

        #print model summary
        print(model.summary())

    def load_model(self, path, trained=False):
        ''' 
        loads a saved model from a specified filepath

        path (str): filepath to a saved keras.model folder
        trained (Bool): specify if the loaded model has already been trained

        '''
        # Check if model already exists, (eg from previous instance of load_model or generate_model)
        if hasattr(self, 'model'):
            raise Exception('model already exists for this object, cannot load model')

        self.model = tf.keras.models.load_model(path)
        self.trained = trained
        print ('model loaded')

    def fit(self, epochs=10):
        '''fits data to self.model, generates model history and training time. Defaults to 10 epochs'''

        # raise exception if model does not yet exist
        if not hasattr(self, 'model'):
            raise Exception('model does not exist yet! please call generate_model() or load_model() first')

        print ('Training model...')

        # record start and end time while fitting data to the model.
        start = time()
        self.model_history = self.model.fit(self.X_train, self.y_train, epochs=epochs)
        self.training_time = (time()-start)
        self.trained = True
        print ('Done.')
    
    def evaluate(self):
        ''' evaluates model against the testing dataset'''

        # Raise exception if the model has not yet been generated or fit.
        if not hasattr(self, 'model'):
            raise Exception('model does not exist yet! please call generate_model() or load_model() first')
        if self.trained == False:
            raise Exception('model has not yet been trained, please train the model by calling fit()')

        self.evaluation = self.model.evaluate(self.X_test,  self.y_test)
    
    def predict_results(self):
        '''Returns a dataframe containing predicted and true values from keras.model.predict method.'''
        
        # Raise exception if the model has not yet been generated or fit.
        if not hasattr(self, 'model'):
            raise Exception('model does not exist yet! please call generate_model() or load_model() first')
        if self.trained == False:
            raise Exception('model has not yet been trained, please train the model by calling fit()')

        # perform predictions on x_test
        print ('generating predictions...')
        self.predictions = pd.DataFrame(self.model.predict(self.X_test))

        # open empty dataframe
        results = pd.DataFrame()

        # find column of max for each row and make new column for 'P' Prediction
        results['P'] = self.predictions.idxmax(axis = 1)

        # reset index of y_test and append actual encoded values as 'A', 
        actual = self.y_test.reset_index(drop=True)
        results['A'] = actual

        self.results = results
        return results

    def confusion_matrix(self):
        '''Confusion matrix of size equal to the amount of possible predictions. Returns a pandas dataframe.'''

        # predict results if results have not yet been calculated
        if not hasattr(self, 'results'):
            self.predict_results()
            
        # calculate a frequency value for each P/A pair.
        freq = self.results.groupby(self.results.columns.tolist(), as_index=False).size()

        # Convert frequency dataframe to np array
        results_array = freq.to_numpy()

        # Generate matrix grid (n x n)
        size = self.num_catagories
        mtx = {}
        for i in range(size):
            for x in range(size):    
                mtx[i] = {i:np.NaN}
        mtx = pd.DataFrame(mtx)

        # iterate over the np array, use the Predicted and Actual values as coordinates to insert the respective size value
        # any missing values will remain as NaN. once filled, replace NaN with 0.
        # matrix is represented with Predictions on the x axis (columns) and Actual values on the y axis (rows)
        for item in results_array:
            mtx[item[0]][item[1]] = item[2]
        mtx = mtx.replace(np.NaN, 0)

        self.confusion_mtx = mtx
        return mtx

    def weighted_confusion_matrix(self):
        '''Applies adjustments to the values in the confusion matrix based on the relative frequency of classification types in the testing data.'''

        # generate confusion matrix if it has not yet been generated
        if not hasattr(self, 'confusion_mtx'):
            self.confusion_matrix()

        # New dataframe for groupby function
        counts = pd.DataFrame()

        # Pass in data and apply groupby, store size values
        counts['y_test'] = self.y_test.reset_index(drop=True)
        counts = counts.groupby('y_test').size()

        # Operations for calculation of adjustment factor for each classification type
        total_count = len(self.y_test)
        adjustment_factor = counts/total_count

        # apply the adjustment factor along rows
        weighted_mtx = self.confusion_mtx.multiply(adjustment_factor, axis = 'index')
        
        self.weighted_confusion_mtx = weighted_mtx
        return weighted_mtx

    def compute_accuracy(self):
        '''compute keras.metrics.accuracy, return accuracy value'''

        # predict results if results have not yet been calculated
        if not hasattr(self, 'results'):
            self.predict_results()

        acc = tf.keras.metrics.Accuracy()
        acc.update_state(self.results['P'], self.results['A'])
        self.accuracy = acc.result().numpy()
    
    def compute_precision(self):
        '''
        extracts the diagonal from the confusion matrix and divides by the sum of the rows. 
        returns a list of precisions for each class in the data.
        '''
        # generate confusion matrix if it has not yet been generated
        if not hasattr(self, 'confusion_mtx'):
            self.confusion_matrix()

        self.precision = np.diag(self.confusion_mtx) / np.sum(self.confusion_mtx, axis = 0)

    def compute_recall(self):
        '''
        extracts the diagonal from the confusion matrix and divides by the sum of the columns. 
        returns a list of recalls for each class in the data.
        '''
        # generate confusion matrix if it has not yet been generated
        if not hasattr(self, 'confusion_mtx'):
            self.confusion_matrix()

        self.recall = np.diag(self.confusion_mtx) / np.sum(self.confusion_mtx, axis = 1)

    def compute_f1(self):
        '''compute f1 score from the mean of precision and recall '''

        # compute recall and precision if they have not already been computed.
        if not hasattr(self, 'precision'):
            self.compute_precision()
        if not hasattr(self, 'recall'):
            self.compute_recall()

        # calculate the harmonic mean of the mean of all recall and precision.
        rec_mean = self.recall.mean()
        prec_mean = self.precision.mean()
        self.f1_score = 2*((prec_mean*rec_mean)/(prec_mean+rec_mean))

    def save_model(self, path):
        '''
        saves the trained or partially trained model (warning large file size)
        saves the model architecture summary as a .txt file
        '''
        # raise exception if models does not yet exist
        if not hasattr(self, 'model'):
            raise Exception('model does not exist yet! please call generate_model() or load_model() first')

        # save model architecture as .txt
        with open(f'{path}/{self.name}_modelsummary.txt', 'w') as f:
            self.model.summary(print_fn=lambda x: f.write(x + '\n'))

        # save the model
        self.model.save(f'{path}/{self.name}_model')

    def save_model_data(self, path):
        '''
        saves the metrics, confusion matricees and training history.
        '''
        # write a report string including metrics and save as txt file.
        report_string = f"Model metrics\n——————————————————————————————\nAccuracy: {self.accuracy}\nPrecision: {self.precision.mean()}\nRecall: {self.recall.mean()}\nF1 Score: {self.f1_score}"
        with open(f'{path}/{self.name}_report.txt', 'w') as f:
            f.write(report_string)
            f.close()

        # save confusion matrix as csv files
        self.confusion_mtx.to_csv(f'{path}/{self.name}_confmtx.csv')
        self.weighted_confusion_mtx.to_csv(f'{path}/{self.name}_wconfmtx.csv')

        # save the training history as csv
        history = pd.DataFrame(self.model_history.history)
        history['epoch'] = np.arange(len(history))
        history['epoch'] = history['epoch'].apply(lambda x: x + 1)
        history.to_csv(f'{path}/{self.name}_trainingHistory.csv')
