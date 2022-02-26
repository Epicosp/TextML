#imports
import requests
import json
import time

def core_api_call(api_key,query):
    '''
    Make API calls to the core data api (https://core.ac.uk/).

    Query the database by abstract search, returns json object 

    api_key (string) = key for accessing database

    query (string) = payload for database search
    '''

    payload = {"abstract":str(query)}

    # payload in json format
    payload_json = json.dumps(payload)

    url = 'https://api.core.ac.uk/v3/recommend?apiKey='+api_key
    
    #return api call in json format
    response = requests.post(url,payload_json).json()

    return response