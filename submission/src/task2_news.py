import pandas as pd
import numpy as np

from helper_functions import parse_html_file,read_filelist
import task1_languages as t1

import os
import json

import logging

import text_preprocessing as tp
from multiprocessing import Pool
import pickle


logFormatter = '%(asctime)s - %(levelname)s - %(message)s'
logging.basicConfig(format=logFormatter, level=logging.DEBUG)
logger = logging.getLogger(__name__)

### ASSET LOCATIONS
newsy_words = "../assets/newsy_words.pickle"
non_newsy_words = "../assets/non_newsy_words.pickle"


### NECESSARY FUNCTIONS

# List a set of newsy_words

# This list can be further expanded by adding top unique words
# found across news articles


def extract_features(html,newsy_words,non_newsy_words,**kwargs):
    '''
    Looks up newsy and no-newsy keywords from the list.
    Return the total word count and newsy counts
    Provide the argument section to extract features corresponding
    to that section. If no section is provided, all_text is used'''
    try:
        section = kwargs['section']
    except KeyError as e:
        section = 'all_text'
   
    if section == 'url':
        import re
        
        news_ctr = 0
        no_news_ctr = 0
        for x in newsy_words:
            regx = re.compile(f'[^a-z]{x}[^a-z]|^{x}[^a-z]|[^a-z]{x}$')
            if regx.search(html['url']):
                news_ctr+=1
        for x in non_newsy_words:
            regx = re.compile(f'[^a-z]{x}[^a-z]|^{x}[^a-z]|[^a-z]{x}$')
            if regx.search(html['url']):
                no_news_ctr+=1     
        r_d = {section+'_num_nw': news_ctr,
               section+'_num_no_nw': no_news_ctr
              }
        return r_d
    # If not url, pre-process text and contnue
    try:
        text = tp.preprocess(html[section])
    except Exception as e:
        print(e)
        text = ""
    
    nw = set(text).intersection(newsy_words)
    no_nw = set(text).intersection(non_newsy_words)   
    
    r_d = {section+'_num_nw':len(nw),
           section+'_num_no_nw':len(no_nw),
           section+'_tot_words' : len(text)
          }
    
    try:
        if kwargs['return_text']:
            r_d.update({section+'_tokens': text})
    except KeyError as e:
        pass
    
    return r_d
 

def aggregate_features(html):
    
    newsy_words = ['news','daily','desk','breaking','archive','times','report','network','post']
    non_newsy_words = ['recipe','horoscope','lottery','advertisement','ad']

#     #Pickle newsy_words
#     with open("../assets/newsy_words.pickle","rb") as p:
#         newsy_words = pickle.load(p)
#     with open("../assets/non_newsy_words.pickle","wb") as p:
#         pickle.dump(non_newsy_words, p)

    # Implement this if the file has to be read from 
    # disk. Currently, html dict is passed
    # html = parse_html_file(fname)
    
    
    html.update(extract_features(html,newsy_words,non_newsy_words,section='url'))
    html.update(extract_features(html,newsy_words,non_newsy_words,section='site_name'))
    html.update(extract_features(html,newsy_words,non_newsy_words,section='title'))
    html.update(extract_features(html,newsy_words,non_newsy_words,section='all_text',return_text=True))
    
    
    # Number of links in the article
    html.update({'num_links' : len(html['links'])})
    
    return html


def news(**kwargs):
    '''
    Can be imported to a different module and
    take source folder as input to print
    the output json with mentioned category
    '''
  
    # Call the languages module to get only EN and RU articles
    # Enable full_path parameter for later manipulations
    try:
        lang_json,df_parsed = t1.languages(kwargs['path'],full_path=True,return_parsed_df=True)
    except KeyError as e:
        logger.error("Source_dir not provided correctly")
        pass
    
    file_list = []
    for f in lang_json:
        file_list.extend(f['articles'])
    
    # Subset the data frame to only consider EN and RU
    df_parsed = df_parsed[df_parsed.fname.isin(file_list)]
    
    
    #### COMPUTE STRUCTURAL FEATURES OF THE FILE
    f = list(df_parsed.html_dict)
    with Pool(7) as pool:
        results = pool.map(aggregate_features,f)
    
    
    # Convert to dataframe and create summed up features

    n_feats = pd.DataFrame(results)

    n_feats['html_dict'] = results

    n_feats['fname'] = df_parsed['fname']
    n_feats['news_keys'] = n_feats['title_num_nw'] + n_feats['site_name_num_nw'] + n_feats['url_num_nw']
    n_feats['not_news_keys'] = n_feats['title_num_no_nw'] + n_feats['site_name_num_no_nw'] + n_feats['url_num_no_nw']
    
    
    # Get the filenames. Use the fullpath parameter
    try:
        full_path = kwargs['full_path']
    except KeyError as e:
        logger.info('full_path parameter not set.')
        full_path = False
    
    if full_path:
        news_articles = list(n_feats[(n_feats['not_news_keys'] <= n_feats['news_keys'])].fname)
    else:
        news_articles = list(n_feats[n_feats['not_news_keys'] <= n_feats['news_keys']].fname.map(os.path.basename))
    
    # Prepare the json output
    out_dict = {"articles":news_articles}
    
    return out_dict,n_feats

if __name__ == "__main__":
    
    logFormatter = '%(asctime)s - %(levelname)s - %(message)s'
    logging.basicConfig(format=logFormatter, level=logging.DEBUG)
    logger = logging.getLogger(__name__)


    import sys
    len(sys.argv)
    if(len(sys.argv)) > 1:
        path = sys.argv[1]
        
        #CALL THE COMPUTATION
        out_dict,n_feats = news(path)
        print(out_dicts)
    else :
        print("Provide source_dir")

