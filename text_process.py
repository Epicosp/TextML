import pandas as pd
import numpy as np
import re
import tensorflow_text as tf_text
from nltk.tokenize import sent_tokenize
import string

def to_data_frame(csv_file):
    '''Takes a csv object, path to csv or json and returns a pandas dataframe'''
    data = pd.read_csv(csv_file)
    data_frame = pd.DataFrame(data)
    return data_frame


def english_papers (data_frame):
    '''uses a dataframe generated from a core_data api response and filters the results by English'''
    for x, paper in enumerate(data_frame['language'].tolist()):
        if 'English' or 'en' not in paper:
            print ('English')#data_frame = data_frame.drop(index = x)
    data_frame = data_frame.reset_index(drop=True)
    return data_frame


def text_clean(text_data):
    '''
    function for cleaning and tokenizing text from scientific papers, using nltk sent_tokenize.

    normalizes to utf8

    slices to remove references

    removes digits

    tokenizes by sentence

    removes unweildly tokens

    returns a list of strings
    '''
    sentences = []
    for text in text_data:
        text = text.lower()
        tf_text.normalize_utf8(text)
        text = text[:text.find('references')] # not a nice way to remove references

        # remove digits
        text = ''.join([i for i in text if not i.isdigit()])

        # tokenize text into list of sentences (retains context)
        tokens = sent_tokenize(text)

        # slice the front of the text off to remove messy document identifiers
        tokens = tokens[20:] 
        for token in tokens:

            # token = token.translate(str.maketrans('', '', string.punctuation)) #used to remove punctuation
            if len(token) > 30 and token.count('.') < 5 and token.count(',') < 6:
                sentences.append(token)
    return sentences


def remove_hyperlinks(data_frame):
    '''removes http strings'''
    data_frame = data_frame.replace('http\S+', '', regex=True)
    return data_frame


