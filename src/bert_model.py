#imports
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text
import pandas as pd
import numpy as np
from time import time

class BertModel:
    def __init__(self, X_train, y_train, X_test, y_test, num_catagories):
        '''
        constructs a tf.keras neural network using bert model for text classification
        X_test: pandas.core.series.Series
        y_test: pandas.core.series.Series
        X_train: pandas.core.series.Series
        y_train: pandas.core.series.Series
        num_catagories: int
        '''
        self.X_test = X_test
        self.y_test = y_test
        self.X_train = X_train
        self.y_train = y_train
        self.num_catagories = num_catagories

    def generate_model(self):
        '''
        NLP architechture consisting of BERT (v4) preprocessing and encoder,
        an input layer with neurons equal to the BERT output length,
        A dropout layer and an output layer with neurons equal to the amount of training catagories.
        
        returns a model ready for training data and prints model architecture to console
        '''
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

    def load_model(self, path):
        ''' 
        loads a pretrained model from a specified filepath

        path: str
        filepath to a saved keras.model folder

        '''
        self.model = tf.keras.models.load_model(path)
        print ('model loaded')

    def fit(self):
        '''
        compiles and fits data to self, evaluates data on test data.
        generates model history and training time.
        '''
        print ('Training model...')

        # record start and end time while fitting data to the model.
        start = time()
        self.model_history = self.model.fit(self.X_train, self.y_train, epochs=10)
        self.training_time = (time()-start)
        print ('Done.')
    
    def evaluate(self):
        ''' evalueates model agains the testing dataset'''
        evaluation = self.model.evaluate(self.X_test,  self.y_test)
        self.evaluation = evaluation
        return evaluation
    
    def predict_results(self):
        '''Returns a dataframe containing predicted and true values from keras.model.predict object.'''
        
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

        # calculate a frequency value for each P/A pair.
        freq = self.results.groupby(self.results.columns.tolist(), as_index=False).size()

        # Convert to frequency dataframe np array
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
        acc = tf.keras.metrics.Accuracy()
        acc.update_state(self.results['P'], self.results['A'])
        self.accuracy = acc.result().numpy()
    
    def compute_precision(self):
        '''
        extracts the diagonal from the confusion matrix and divides by the sum of the rows. 
        
        returns a list of precisions for each class in the data.
        '''
        self.precision = np.diag(self.confusion_mtx) / np.sum(self.confusion_mtx, axis = 0)

    def compute_recall(self):
        '''
        extracts the diagonal from the confusion matrix and divides by the sum of the columns. 
        
        returns a list of recalls for each class in the data.
        '''
        self.recall = np.diag(self.confusion_mtx) / np.sum(self.confusion_mtx, axis = 1)