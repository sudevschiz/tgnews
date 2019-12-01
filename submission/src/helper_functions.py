import pandas as pd
import numpy as np

from multiprocessing import Pool
import glob

import text_preprocessing as tp

import logging
import os
import sys
import json


### LOGGER
logFormatter = '%(asctime)s - %(levelname)s - %(message)s'
logging.basicConfig(format=logFormatter, level=logging.DEBUG)
logger = logging.getLogger(__name__)

### ASSETS
EN_VEC = "../assets/wiki-news-300d-500k.vec"

### GLOBAL VARIABLES
N_CORES = 8
# Figure how to limit this
MAX_RAM = 16


### NECESSARY FUNCTIONS
def read_filelist(folder_path,**kwargs):
    '''Return the relative file path of all the html files 
    which are present in the path
    
    By defauly, the search is recursive and extends to all the folders
    present in the folder_path provided.
    
    Pass `recursive=False` for fetch only the current
    directory
    
    Pass the file extension desired as 'file_type'.
    Eg: file_type = html (Default is 'htm')
    
    '''
    try:
        recursive = kwargs['recursive']
        if not isinstance(recursive,bool):
            logger.info('Recursive option not passed correctly. Defaulting to True')
            recursive = True
    except KeyError as e:
        recursive = True
    
    try:
        file_type = kwargs['file_type']
        if not isinstance(file_type,str):
            logger.info("Defaulting to 'htm'")
            file_type = 'htm'
    except KeyError as e:
        logger.info("Defaulting to 'htm*'")
        file_type = 'htm*'
    r_path = os.path.join(folder_path, "**/*."+file_type)
    file_list = [f for f in glob.glob(r_path, recursive=recursive)]
    return file_list


def get_soup(file):
    '''Return the BeautifulSoup object of the html file provided'''
    from bs4 import BeautifulSoup
    
    with open(file,'r') as file_ptr:
        soup = BeautifulSoup(file_ptr,'lxml')
    return soup


def extract_meta(soup):
    '''This is a specific function for the type of file TG has
    provided in the samples. The look-up values are based on that
    
    If those fields are not found (as in the case of a generic html)
    empty values are returned in those cases'''
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
    '''Takes the soup objects and the tag name as inputs,
    returns all the text in that tag concatenated 
    together'''
    
    if tag == 'all':
        text = soup.text.strip()
    else:
        p_contents = soup.find_all(tag)
        text = ""
        for p in p_contents:
            text = text + " " + p.getText()
    return text


def sanitize_text(text):
    import re
    sane_text = re.sub(r'^https?:\\/\\/.*[\\r\\n]*', '',text, flags=re.MULTILINE)
    sane_text = bytes(sane_text, 'utf-8').decode('utf-8','ignore')
    
    return sane_text


def extract_links(soup,domain=False):
    '''Takes the soup objects and returns all the links'''
    links = [a.get('href') for a in soup.find_all('a', href=True)]

    return links


def parse_html_file(file):
    '''Uses bs4 to get the soup of the file and calls
    the other extraction functions
    TODO : Better html parsers are available'''
    soup = get_soup(file)
    d = extract_meta(soup)
    d['p_text'] = extract_text(soup,'p')
    d['links'] = extract_text(soup,'a')
    d['all_text'] = d['title'] + "\n" + d['p_text']
    d['links'] =extract_links(soup)
    return d


def load_vectors(fname): 
    import io
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
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


def compute_ft_sum(textlist,ft_dict=None,d=300,**kwargs):
    '''ft_dict should be loaded to memory before-hand!
    If ft_dict is not passed as it is, the relative path
    of the vector has to be provided in the argument 
    'vec_loc'
    TODO : Need better implemenation of this.
    '''
    
    if not ft_dict:
        try:
            # Load the FastText vector. MUST IMPROVE SPEEDS
            from time import time
            t_start = time()
            _,_,ft_dict = load_vectors(EN_VEC)
            logger.info(f"FT_Vector loaded in {time()-t_start} seconds")
        except FileNotFoundError as e:
            logger.error("Vector not found")
    
    ret_vec = np.array([0]*d)
    for key in textlist:
        vec = get_ft_vec(key,ft_dict)
        # Vector addition of token embeddings
        ret_vec = ret_vec + vec
    return list(ret_vec)