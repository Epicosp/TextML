#imports
import requests
import json

def core_api_call(api_key,query):
    '''
    Make API calls to the core data api (https://core.ac.uk/).

    Query the database by abstract search, returns json object 

    api_key (string) = key for accessing database

    query (string) = payload for database search
    '''

    payload = {'abstract':str(query)}

    # payload in json format
    payload_json = json.dumps(payload)

    url = 'https://api.core.ac.uk/v3/recommend?apiKey='+api_key
    
    #return api call in json format
    response = requests.post(url,payload_json).json()

    return response

def ask_user():
    '''
    Asks for user input, requests two or more topics to train an NLP model on.

    Returns dict, k='input' : v='neumerical encoding'
    '''
    input_list = []
    while True:
        if len(input_list) < 2:
            user_in = input(f'Select a topic to train on: ')
            input_list.append(user_in)
        else:
            user_in = input('Add another topic? (type \'done\' to continue): ')
            if user_in.lower() == 'done':
                break
            else:
                input_list.append(user_in)


    
    input_dict = {key: value for value, key in enumerate(input_list)}
    return input_dict
