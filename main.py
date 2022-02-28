import os
from api_call import core_api_call
import pandas as pd
import text_process as tp

def main():
    # use API key stored in environment variables
    CORE_API_KEY = os.getenv('CORE_API_KEY')

    # pick 10 subjects to train on
    queries = {'Blockchain':0,
    'Cryptocurrency':1,
    'Genetic engineering':2,
    'Machine learning':3,
    'Nanotechnology':4,
    'Quantum computing':5,
    'Robotics':6,
    'Social engineering':7,
    'Space exploration':8,
    'Virtual reality':9}

    # dataframe to append processed text
    all_text = pd.DataFrame()

    # loop through queries and process responses 
    for key, value in queries.items():
    
        # use local files in preference over API calls, if files dont exist, call api and save data locally.
        try:
            data = pd.DataFrame(pd.read_csv(f'API_responses/{key}.csv'))
        except:
            data = pd.DataFrame(core_api_call(CORE_API_KEY, key))
            print (f'Acessing CORE Database for information about {key}')
            data.to_csv(f'API_responses/{key}.csv', index = False)

        #process response
        data = tp.english_papers(data, 'English')
        data = tp.remove_hyperlinks(data)
    
        # tokenize text into sentences and convert to dataframe
        data = pd.DataFrame(tp.text_clean(data['fullText']))
    
        # add column for encoding
        data['Code'] = value

        # rename columns
        data.rename(columns = {0:'Text'}, inplace=True)

        # append to final dataframe
        all_text = all_text.append(data, ignore_index = True)

    # save dataframe to csv
    all_text.to_csv('processed_text/NLP_data_test.csv', index = False)

if __name__ == "__main__":
    main()
