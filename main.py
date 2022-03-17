import os
import src.api_call as ac
import pandas as pd
import src.text_process as tp
from sklearn.model_selection import train_test_split
import src.bert_model as bm
import src.model_evaluation as me
from pathlib import Path
import seaborn as sns
from matplotlib import pyplot as plt

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

    #train/test split
    x_train,x_test,y_train,y_test = train_test_split(all_text['Text'],all_text['Code'])

    #generate a model
    print ('generating model...')
    model = bm.generate_model(len(queries))

    #train model

    model_history, train_time, eval = bm.compile_fit_evaluate(model, x_train, y_train, x_test, y_test)

    # used trained model to predict y_test values
    predictions = model.predict(x_test)

    #generate confusion matrix, save to local file
    print ('Evaluating model...')
    confusion_mtx = me.confusion_matrix(predictions, y_test)
    sns.heatmap(confusion_mtx, annot=True)
    plt.savefig(f'{model_name}/confusion_matrix.png', dpi = 400)

    # save text and model information
    me.save_model_data(model,eval,model_history,model_name)

if __name__ == "__main__":
    main()
