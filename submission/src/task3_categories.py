import pandas as pd
import numpy as np

import lightgbm as lgb

from bs4 import BeautifulSoup

from multiprocessing import Pool
import glob

import text_preprocessing as tp

import task2_news as t2

import logging
import os
import sys
import json


### LOGGER
logFormatter = '%(asctime)s - %(levelname)s - %(message)s'
logging.basicConfig(format=logFormatter, level=logging.DEBUG)
logger = logging.getLogger(__name__)

### ASSETS 
ru_vec = ""
en_vec = "../assets/wiki-news-300d-500k.vec"

model_file = "../assets/models/lgb_model_all_data.txt"

### RELEVANT FUNCTIONS FOR THIS TASK
def labelize_numeric(arr):
    '''Change the numeric label output from the 
    model to human readable names'''
    d = {0:'society',
         1:'economy',
         2:'technology',
         3:'sports',
         4:'entertainment',
         5:'science',
         6:'other'}
    return d[arr]


def predict_label(df_test,model):
    '''Make the prediction using LightGBM prediction
    method. Return the class with maximum probability
    
    TODO : There should be better logic in choosing
    the final label'''
    # TODO : Implement other logic for the predction
    y_pred = model.predict(df_test)
    max_p_index = np.argmax(y_pred,axis=1)
    
    label = [labelize_numeric(i) for i in max_p_index]
    return label


def prepare_output(y_pred,file_list):
    '''Helper function to prepare the output in the TG way
    Takes the predictions and the correspoding file names
    and returns a grouped json object'''
    fname = [os.path.basename(p) for p in file_list]
    _d = pd.DataFrame({'html_fname' : fname,'label' : y_pred})
    out_dict = dict(_d.groupby('label').apply(lambda x : [f for f in x['html_fname']]))
    out_json = json.dumps(out_dict)
    
    return out_json


def categories(*args):
    '''Make calls in the logical predict way.
    Can be imported to a different module and
    take source folder as input to print
    the output json with mentioned category'''
    try : 
        # Get file_list
        file_list = read_filelist(args[0])
    except IndexError as e:
        print('ERR:Source directory missing')
        exit()
    
    # TODO : Extract only News articles (Task 2)
    # This will in internally select only EN and RU
    # articles
    
    en_ru_news = t2.news(file_list)
  
    # Load the predictive model
    model = lgb.Booster(model_file=model_file)
    
    
    #### START EXECUTION ON FILES
    with Pool(N_CORES) as pool:
        # Pass only the news articles to parse function
        # Ideally, the parsed files from task2 should be
        # reused
        html_list = pool.map(parse_html_file,en_ru_news)
    
    # Create a list of text only
    test_list = [d['all_text'] for d in html_list]
    
   
    with Pool(N_CORES) as pool:
        # Compute FT vector
        vector = pool.map(compute_ft_sum,en_ru_list)
    
    # Convert to datafram and predict label
    df_test = pd.DataFrame(vector)
    y_pred = predict_label(df_test,model)

    # Prepare the json output
    output = prepare_output(y_pred,file_list)
    
    return output

if __name__ == "__main__":
    if(len(sys.argv)) > 1:
        path = sys.argv[1]
        
        #CALL THE COMPUTATION
        print(categories(path))
    else :
        print("Provide source_dir")

