import os
import time
from api_call import core_api_call
import pandas as pd

def main():
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


# use API key stored in environment variables
if __name__ == "__main__":
    main()


# save response as CSV
'''
saves a json response object as a csv file
api_response (json object)
'''
def save_response(api_response, name):
    df = pd.DataFrame(api_response)
    df.to_csv('API_responses/'+str(name)+'.csv', index = False)
