#imports
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text
import pandas as pd
import numpy as np
from time import time

class BertModel:
    def __init__(self, X_text, y_test, X_train, y_train, num_catagories):
        '''
        constructs a tf.keras neural network using bert model for text classification
        X_test: pandas.core.series.Series
        y_test: pandas.core.series.Series
        X_train: pandas.core.series.Series
        y_train: pandas.core.series.Series
        num_catagories: int
        '''
        self.X_text = X_text
        self.y_test = y_test
        self.X_train = X_train
        self.y_train = y_train
        self.num_catagories = num_catagories
        self.model = ""
        self.evaluation = ""

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
        self.evaluation = self.model.evaluate(self.X_test,  self.y_test)





