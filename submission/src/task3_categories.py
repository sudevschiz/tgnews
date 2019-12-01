import pandas as pd
import numpy as np

import lightgbm as lgb

from multiprocessing import Pool
import glob

from helper_functions import compute_ft_sum,load_vectors
import task2_news as t2

import logging
import os
import sys


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
    and returns a dictionary'''
    fname = [os.path.basename(p) for p in file_list]
    _d = pd.DataFrame({'html_fname' : fname,'label' : y_pred})
    out_dict = dict(_d.groupby('label').apply(lambda x : [f for f in x['html_fname']]))
    
    return out_dict


def categories(**kwargs):
    '''Make calls in the logical predict way.
    Can be imported to a different module and
    take source folder as input and returns 
    the dictionary with mentioned category
    as output'''
    try : 
        # Get file_list
        html = kwargs['html']
        logger.info('Parsed html files are passed.')
    except KeyError as e:
        logger.info('Did not received parsed html. Trying file_list')
        try : 
            path = kwargs['path']
        except KeyError as e:
            logger.error("Neither html nor file_list passed. Check inputs")
            pass
    
    if path:
        out_dict,n_feats = t2.news(path = path)
        html = n_feats['html_dict']
  
    if html.any():   
        #### START EXECUTION ON FILES
        
#         tokens = [x['all_text_tokens'] for x in html]
#         with Pool(N_CORES) as pool:
#             # Compute FT vector
#             vector = pool.map(compute_ft_sum,tokens)
        
        _,_,ft_dict = load_vectors(en_vec)
        vector = [compute_ft_sum(h['all_text_tokens'],ft_dict) for h in html]
        # Additional feature creations here
        
        ## ###
    
    # PREDICTION
    model = lgb.Booster(model_file=model_file)
   
    # Convert to dataframe and predict label
    df_test = pd.DataFrame(vector)
    y_pred = predict_label(df_test,model)

    # Prepare the json output
    output = prepare_output(y_pred,n_feats['fname'])
    
    return output

if __name__ == "__main__":
    if(len(sys.argv)) > 1:
        path = sys.argv[1]
        
        #CALL THE COMPUTATION
        print(categories(path))
    else :
        print("Provide source_dir")

