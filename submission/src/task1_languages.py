import pandas as pd
import numpy as np
import random
import glob
import logging
import os
import re
from multiprocessing import Pool

import helper_functions

import pycld2 as cld2

from bs4 import BeautifulSoup
import sys

# Logger
logFormatter = '%(asctime)s - %(levelname)s - %(message)s'
logging.basicConfig(format=logFormatter, level=logging.DEBUG)
logger = logging.getLogger(__name__)
    
### Functions

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

# TODO : Make a good sanitization function @Jun
def sanitize_text(text):
    sane_text = text
    return sane_text

def compute_lang_prob(t):
    top_l = None
    top_l_prob = 0.0
    
    en_prob = 0.0
    ru_prob = 0.0
    try: 
        for l in t[2]:
            if l[2]>top_l_prob:
                top_l_prob = l[2]
                top_l = l[1]
            if l[1] == 'en':
                en_prob = l[2]
            elif l[1] == 'ru':
                ru_prob = l[2]
    except :
        pass

    return {'top_l' : top_l, 'top_l_prob' :top_l_prob ,'en_prob' : en_prob, 'ru_prob' : ru_prob}


def detect_langage(text,method = 'cld2'):
    # Pass the 'method' parameter for deferent
    # models. 
    # Valid params = [cld2,langdetect,polyglot]
    
    ## Encode to utf-8
    text = text.encode('utf-8').decode("utf-8", "ignore")
    
    try:
        if method == 'cld2':
            # Pass to cld2
            result = cld2.detect(text, bestEffort=False)
        elif method == 'langdetect':
            ### TODO : return values properly
            result = detect_langs(text)
        elif method == 'polyglot':
            ### TODO : implement polyglot
            result = tuple()
        else:
            result = tuple()
    except:
        result = tuple()
    
    # Now, compute the probabilities
    _p = compute_lang_prob(result)
    return _p


def detect_distributed(file):
    soup = get_soup(file)
    d = extract_meta(soup)
    d['p_text'] = extract_text(soup,'p')
    d['all_text'] = sanitize_text(d['title'] + "\n" + d['p_text'])
    d.update(detect_langage(d['all_text']))
    
    return d


def label_final_lang(df_prob,prob=0.95):
    # For now, extract the cases where model was > 95% sure
    en_articles = list(df_prob[df_prob['en_prob']>=prob]['fname'])
    ru_articles = list(df_prob[df_prob['ru_prob']>=prob]['fname'])
    return en_articles,ru_articles


def prepare_output(lang_code,article_list):
    #TODO : Make sure lang_code is a valid 
    #       ISO 639-1 two-letter language code
    d = {"lang_code" : lang_code,"articles":article_list}
    return d


def languages(path):
    file_list = read_filelist(path)
#     logger.info(f'Number of files : {len(file_list)}')
    
    with Pool() as pool:
        results = pool.map(detect_distributed, file_list)
    
    df_prob = pd.DataFrame(results)
    df_prob['fname'] = [os.path.basename(f) for f in file_list]
    
    # For now, extract the cases where model was > 95% sure
    en_articles,ru_articles = label_final_lang(df_prob,prob=0.95)
    
    output = [prepare_output("en",en_articles),prepare_output("ru",ru_articles)]
    
    return output

if __name__ == "__main__":
    if(len(sys.argv)) > 1:
        path = sys.argv[1]
        logger.info('SOURCE_DIR : '+ path)
        print(languages(path))