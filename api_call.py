#imports
import time
import os
import requests
import json
import pandas as pd


# Make API call, query the database by abstract search.
def core_api_call(api_key,query):
    payload = {"abstract":str(query)}

    # payload in json format
    payload_json = json.dumps(payload)

    url = 'https://api.core.ac.uk/v3/recommend?apiKey='+api_key
    
    #return api call in json format
    response = requests.post(url,payload_json).json()
    return response



# save response as CSV
def save_response(api_response, name):
    df = pd.DataFrame(api_response)
    df.to_csv('API_responses/'+str(name)+'.csv', index = False)


# use API key stored in environment variables
if __name__ == "__main__":
    CORE_API_KEY = os.getenv('CORE_API_KEY')
    queries = ['Blockchain',
    'Cryptocurrency',
    'Genetic engineering',
    'Machine learning',
    'Nanotechnology',
    'Quantum computing',
    'Robotics',
    'Social engineering',
    'Space exploration',
    'Virtual reality']

    # loop through queries, call api and save data for each
    for q in queries:
        data = core_api_call(CORE_API_KEY, q)

        # sleep 1 second to allow time for database to return a response.
        time.sleep(1)
        save_response(data, q)
    
# this is on dev branch
         

