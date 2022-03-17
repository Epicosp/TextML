#imports
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text
import pandas as pd
import numpy as np
from time import time

def generate_model(num_catagories):
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
    layer = tf.keras.layers.Dense(num_catagories, activation='sigmoid', name='output')(layer)
    model = tf.keras.Model(inputs=[text_input], outputs = [layer])

    #print and return model
    print (model.summary())
    return model



def compile_fit_evaluate (model, x_train, y_train, x_test, y_test):
    '''
    compiles and fits data to an already generated model, evaluates data on test data.
    returns model history, training time and evaluation.
    '''
    print ('Training model...')
    model.compile(optimizer="adam", loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=["accuracy"])
    start = time()
    model_history = model.fit(x_train,y_train,epochs=10)
    train_time = (time()-start)
    evaluation = model.evaluate(x_test,  y_test)
    return model_history, train_time, evaluation





