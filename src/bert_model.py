#imports
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text
import pandas as pd
import numpy as np
from time import time

class BertModel:
    def __init__(self, X_train, y_train, X_test, y_test, num_catagories, model_name='BertModel'):
        '''
        constructs a tf.keras neural network using bert model for text classification
        model_name: string
        X_test: pandas.core.series.Series
        y_test: pandas.core.series.Series
        X_train: pandas.core.series.Series
        y_train: pandas.core.series.Series
        num_catagories: int
        '''
        self.name = model_name
        self.X_test = X_test
        self.y_test = y_test
        self.X_train = X_train
        self.y_train = y_train
        self.num_catagories = num_catagories

    def generate_model(self):
        '''
        NLP architechture consisting of BERT (v4) preprocessing and encoder,
        an input layer with neurons equal to the BERT output length,
        A dropout layer and an output layer with neurons equal to num_catagories.
        
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
        '''fits data to self.model, generates model history and training time.'''

        print ('Training model...')

        # record start and end time while fitting data to the model.
        start = time()
        self.model_history = self.model.fit(self.X_train, self.y_train, epochs=10)
        self.training_time = (time()-start)
        print ('Done.')
    
    def evaluate(self):
        ''' evaluates model against the testing dataset'''
        self.evaluation = self.model.evaluate(self.X_test,  self.y_test)
    
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

    def compute_f1(self):
        '''compute f1 score from the mean of precision and recall '''

        rec_mean = self.recall.mean()
        prec_mean = self.precision.mean()
        self.f1_score = 2*((prec_mean*rec_mean)/(prec_mean+rec_mean))

    def save_model(self, path):
        '''
        saves the trained or partially trained model (warning large file size)
        saves the model architecture summary as a .txt file
        '''
        # save model architecture as .txt
        with open(f'{path}/{self.name}/modelsummary.txt', 'w') as f:
            self.model.summary(print_fn=lambda x: f.write(x + '\n'))

        # save the model
        self.model.save(f'{path}/{self.name}/model')

    def save_model_data(self, path):
        '''
        saves the metrics, confusion matricees and training history.
        '''
        # write a report string including metrics and save as txt file.
        report_string = f"Model metrics\n -------------------\n Accuracy: {self.accuracy}\nPrecision: {self.precision.mean()}\n Recall: {self.recall.mean()}\n F1 Score: {self.f1_score}"
        with open(f'{path}/{self.name}_report.txt', 'w') as f:
            f.write(report_string)
            f.close()

        # save confusion matrix as csv files
        self.confusion_mtx.to_csv(f'{path}/{self.name}/confusion_matrix.csv')
        self.weighted_confusion_mtx.to_csv(f'{path}/{self.name}/weighted_confusion_matrix.csv')

        # save the training history as csv
        history = pd.DataFrame(self.model_history.history)
        history['epoch'] = np.arange(len(history))
        history['epoch'] = history['epoch'].apply(lambda x: x + 1)
        history.to_csv(f'{path}/{self.name}/model_history.csv')
