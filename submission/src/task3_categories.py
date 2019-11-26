import pandas as pd
import numpy as np

import lightgbm as lgb

from bs4 import BeautifulSoup
import os
from multiprocessing import Pool
import glob

import text_preprocessing as tp

import task1_languages as t1

import json


### ASSETS 

ru_vec = ""
en_vec = "../assets/wiki-news-300d-1M.vec"

model_file = "../assets/models/lgb_model_all_data.txt"

### GLOBALLY LOAD THE FT VECTOR
# Load the FastText vector. MUST IMPROVE SPEEDS


def load_vectors(pretrain_vector_fname): 
    '''TODO : Very time consuming. Needs improvement'''
    import io
    fin = io.open(pretrain_vector_fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split())
    data = {}
    for line in fin:
        tokens = line.rstrip().split(' ')
        data[tokens[0]] = list(map(float, tokens[1:]))
    return n,d,data

_,_,ft_dict = load_vectors(en_vec)


### GLOBAL VARIABLES
N_CORES = 8


### NECESSARY FUNCTIONS

def read_filelist(folder_path = "."):
    r_path = os.path.join(folder_path, "**/*.html")
    file_list = [f for f in glob.glob(r_path, recursive=True)]
    return file_list


def get_soup(file):
    with open(file,'r') as file_ptr:
        soup = BeautifulSoup(file_ptr,'lxml')
    return soup


def extract_meta(soup):
    d = {}
    
    #TODO : Add exception handle to all of this
    try: 
        d['title'] = soup.find("meta",  property="og:title")['content']
    except TypeError as e:
#         logger.error('Title not found')
        d['title'] = ""
    
    try:
        d['url'] = soup.find("meta",  property="og:url")['content']
    except TypeError as e:
#         logger.error('Title not found')
        d['url'] = ""
    
    try:
        d['site_name'] = soup.find("meta",  property="og:site_name")['content']
    except TypeError as e:
#         logger.error('Title not found')
        d['site_name'] = ""
    
    try:
        d['published_time'] = soup.find("meta",  property="article:published_time")['content']
    except TypeError as e:
#         logger.error('Title not found')
        d['published_time'] = ""
    
    try:
        d['description'] = soup.find("meta",  property="og:title")['content']
    except TypeError as e:
#         logger.error('Title not found')
        d['published_time'] = ""
    
    return d

def extract_text(soup,tag = 'all'):
    assert tag in ['all','p','h1']
    if tag == 'all':
        text = soup.text.strip()
    else:
        p_contents = soup.find_all(tag)
        text = ""
        for p in p_contents:
            text = text + p.getText()
    return text

def parse_html_file(file):
    soup = get_soup(file)
    d = extract_meta(soup)
    d['p_text'] = extract_text(soup,'p')
    d['all_text'] = d['title'] + "\n" + d['p_text']
    return d


def load_vectors(pretrain_vector_fname): 
    '''TODO : Very time consuming. Needs improvement'''
    import io
    fin = io.open(pretrain_vector_fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split())
    data = {}
    for line in fin:
        tokens = line.rstrip().split(' ')
        data[tokens[0]] = list(map(float, tokens[1:]))
    return n,d,data


def get_ft_vec(token,ft_dict,d=300):
    try:
        v = np.array(ft_dict[token])
    except KeyError:
        v = np.array([0]*d)
    return v


def compute_ft_sum(textlist,ft_dict=ft_dict,d=300):
    '''ft_dict should be loaded to memory before-hand'''
    ret_vec = np.array([0]*d)
    for key in textlist:
        vec = get_ft_vec(key,ft_dict)
        # Vector addition of token embeddings
        ret_vec = ret_vec + vec
    return list(ret_vec)


def labelize_numeric(arr):
    d = {0:'society',
         1:'economy',
         2:'technology',
         3:'sports',
         4:'entertainment',
         5:'science',
         6:'other'}
    return d[arr]


def predict_label(df_test,model):
    # TODO : Implement other logic for the predction
    y_pred = model.predict(df_test)
    max_p_index = np.argmax(y_pred,axis=1)
    
    label = [labelize_numeric(i) for i in max_p_index]
    return label


def prepare_output(y_pred,file_list):
    fname = [os.path.basename(p) for p in file_list]
    _d = pd.DataFrame({'html_fname' : fname,'label' : y_pred})
    out_dict = dict(_d.groupby('label').apply(lambda x : [f for f in x['html_fname']]))
    out_json = json.dumps(out_dict)
    
    return out_json


def categories(*args):
    try : 
        # Get file_list
        file_list = read_filelist(args[0])
    except IndexError as e:
        print('ERR:Source directory missing')
        exit()
    
    # Load the predictive model
    model = lgb.Booster(model_file=model_file)
    
    # Load the FastText vector. MUST IMPROVE SPEEDS
#     _,_,ft_dict = load_vectors(en_vec)
    
    #### START EXECUTION ON FILES
    with Pool(N_CORES) as pool:
        # Read html text
        html_list = pool.map(parse_html_file,file_list)
    
    # Create a list of text only
    test_list = [d['all_text'] for d in html_list]
    # Create a list of filenames only
    
    # TODO : Extract only EN and RU articles (Task 1)
    # Make sure that file_list is kept on track
    
    # TODO : Extract only News articles (Task 2)
    # Make sure that file_list is kept on track
    en_ru_list = test_list
    # Predict the categories
    with Pool(N_CORES) as pool:
        # Use partial function to add the fixed parame
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

