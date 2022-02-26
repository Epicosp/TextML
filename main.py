import os
import time
from api_call import core_api_call
import pandas as pd
import text_process as tp

def main():
    # use API key stored in environment variables
    CORE_API_KEY = os.getenv('CORE_API_KEY')

    # pick 10 subjects to train on
    queries = {'Blockchain':0,
    'Cryptocurrency':1}
    # 'Genetic engineering':2,
    # 'Machine learning':3,
    # 'Nanotechnology':4,
    # 'Quantum computing':5,
    # 'Robotics':6,
    # 'Social engineering':7,
    # 'Space exploration':8,
    # 'Virtual reality':9}

    # dataframe to append processed text to and save.
    all_text = pd.DataFrame()

    # loop through queries and process responses 
    # maybe use a try except block and try to use locally stored files before querying database
    #for key, value in queries.items():
    data = core_api_call(CORE_API_KEY, 'Blockchain')

    #process response
    data = pd.DataFrame(data)
    print (data['language'])
    data = tp.english_papers(data)
    #text_data = tp.remove_hyperlinks(text_data)
    print (data.head())
        # tokenize text into sentences and convert to dataframe
    data = pd.DataFrame(tp.text_clean(data['fullText']))
    print (data.head())
        # add column for encoding
    data['Code'] = 0

        # rename columns
    data.rename(columns = {0:'Text'}, inplace=True)

    all_text.append(data)

    all_text.to_csv('processed_text/NLP_data_test.csv', index = False)

if __name__ == "__main__":
    main()
