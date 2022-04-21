import os
import src.api_call as ac
import pandas as pd
import src.text_process as tp
from sklearn.model_selection import train_test_split
from src.bert_model import BertModel
from pathlib import Path

def main():
    # use API key stored in environment variables
    CORE_API_KEY = os.getenv('CORE_API_KEY')

   # ask for user input
    model_name = input(f'name the model to be trained (this will make a new directory): ')
    
    #make directory using model name to store resources
    try:
        Path(model_name).mkdir(parents=True, exist_ok=False)
        Path(f'{model_name}/raw_data').mkdir(parents=True, exist_ok=False)
    except:
        pass

    # ask for user input
    queries = ac.ask_user()

    # dataframe to append processed text
    all_text = pd.DataFrame()

    # loop through queries and process text data
    for key, value in queries.items():
    
        # use local files in preference over API calls, if files dont exist, call api and save data locally.
        try:
            data = pd.DataFrame(pd.read_csv(f'{model_name}/raw_data/{key}.csv'))
        except:
            data = pd.DataFrame(ac.core_api_call(CORE_API_KEY, key))
            print (f'Acessing CORE Database for information about {key}')
            data.to_csv(f'{model_name}/raw_data/{key}.csv', index = False)

        # process response
        print (f'cleaning {key} data...')
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

    #drop duplicates from final data and save to csv for inspection
    all_text.drop_duplicates(inplace = True)
    all_text.to_csv(f'{model_name}/clean_data.csv', index = False)

    # train/test split
    x_train,x_test,y_train,y_test = train_test_split(all_text['Text'],all_text['Code'])

    # init BertModel class
    model = BertModel(
        X_train=x_train,
        X_test = x_test,
        y_train = y_train,
        y_test = y_test,
        num_catagories = len(queries),
        model_name = model_name
    )
    model.generate_model()

    # fit model
    model.fit()

    # used trained model to predict y_test values
    model.predict_results()

    # generate confusion matrix
    model.confusion_matrix()
    model.weighted_confusion_matrix()

    # generate metrics/scores
    model.compute_accuracy()
    model.compute_precision()
    model.compute_recall()
    model.compute_f1()
    
    #save model and data
    model.save_model(path = Path(model_name))
    model.save_model_data(path = Path(model_name))

if __name__ == "__main__":
    main()
