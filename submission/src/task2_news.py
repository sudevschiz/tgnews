import pandas as pd
import numpy as np

import lightgbm as lgb

from bs4 import BeautifulSoup

from multiprocessing import Pool
import glob

from helper_functions import * #Bad practise

import task1_languages as t1

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


### NECESSARY FUNCTIONS


def prepare_output(y_pred,file_list,threshold=0.5):
    '''Helper function to prepare the output in the TG way
    Takes the predictions and the correspoding file names
    and returns a grouped json object'''
    fname = [os.path.basename(p) for p in file_list]
    _d = pd.DataFrame({'html_fname' : fname,'prob' : y_pred})
    
    l = _d[_d.prob > threshold].html_fname
    out_dict = {"articles":l}
    out_json = json.dumps(out_dict)
    
    return out_json


def news(*args):
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
    
    
    # Load the predictive model
    model = lgb.Booster(model_file=model_file)
    
    
    #### START EXECUTION ON FILES
    with Pool(N_CORES) as pool:
        # Pass only the news articles to parse function
        # Ideally, the parsed files from task2 should be
        # reused
        html_list = pool.map(parse_html_file,file_list)
    
    # Create a list of text only
    test_list = [d['all_text'] for d in html_list]
    # Create a list of filenames only
    
   
    # Predict the categories
    with Pool(N_CORES) as pool:
        # Use partial function to add the fixed parame
        # Compute FT vector
        vector = pool.map(compute_ft_sum,en_ru_list)

    # Convert to dataframe and predict label
    df_test = pd.DataFrame(vector)
    y_pred = model.predict(df_test)
    
    # Prepare the json output
    output = prepare_output(y_pred,file_list)
    
    return output

if __name__ == "__main__":
    if(len(sys.argv)) > 1:
        path = sys.argv[1]
        
        #CALL THE COMPUTATION
        print(news(path))
    else :
        print("Provide source_dir")

