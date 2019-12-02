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


logFormatter = '%(asctime)s - %(levelname)s - %(message)s'
logging.basicConfig(format=logFormatter, level=logging.DEBUG)
logger = logging.getLogger(__name__)



def prepare_output(y_pred,file_list):
    '''Helper function to prepare the output in the TG way
    Takes the predictions and the correspoding file names
    and returns a dictionary'''
    fname = [os.path.basename(p) for p in file_list]
    _d = pd.DataFrame({'html_fname' : fname,'label' : y_pred})
    out_dict = dict(_d.groupby('label').apply(lambda x : [f for f in x['html_fname']]))
    
    return out_dict

def threads(**kwargs):
    '''Make calls to find the threads.
    Source folder as input and returns 
    the dictionary threads as output'''
    try : 
        # Get file_list
        html = kwargs['vector']
        logger.info('Vector files are passed.')
    except KeyError as e:
        logger.info('Did not received parsed html. Trying file_list')
        try : 
            path = kwargs['path']
        except KeyError as e:
            logger.error("Neither html nor file_list passed. Check inputs")
            pass
    
    if path:
        _,df = t2.news(path = path)
        
        out_dict = {}
            
        cat = cat[cat.isna().apply(sum,axis=1) == 0]
            
        X = cat.drop(['label','pubished_time','title'],axis=1)
            
        from sklearn.decomposition import PCA
        pca = PCA(n_components=10, svd_solver='full')
            
        df_cat = pd.DataFrame(pca.fit_transform(X))
            
        logger.info("Variance explained : " + str(pca.explained_variance_ratio_.sum())) 

        # Fit the DBSCAN clustering model
        m = DBSCAN(eps=0.3, min_samples=2)
            
        return output,df_test

if __name__ == "__main__":
    

    if(len(sys.argv)) > 1:
        path = sys.argv[1]
        
        #CALL THE COMPUTATION
        print(threads(path=path))
    else :
        print("Provide source_dir")

