from helper_functions import *

import pycld2 as cld2


### FUNCTIONS FOR LANGUAGE DETECTION
def compute_lang_prob(t):
    '''Returns the top language code identified.
    Also returns the probaility of English and Russian
    (Specific to this problem statement)'''
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
    '''For each piece of text input, this
    function uses the method passed to return
    the detected languages
    Pass the 'method' parameter for different models. 
    Valid params = [cld2,langdetect,polyglot]'''
    
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
    except Exception as e:
        logger.error(e)
        result = tuple()
    
    # Now, compute the probabilities
    _p = compute_lang_prob(result)
    return _p


def detect_distributed(file):
    '''This function calls the process in order.
    Parallizes well.'''
#     soup = get_soup(file)
#     d = extract_meta(soup)
#     d['p_text'] = extract_text(soup,'p')
#     d['all_text'] = sanitize_text(d['title'] + "\n" + d['p_text'])
    d = parse_html_file(file)
    d.update(detect_langage(d['all_text']))
    
    return d


def label_final_lang(df_prob,prob=0.95):
    '''Once the probabilities are calculated,
    prepare the list of EN and RU articles.
    TODO : Ideally this should scale to any language.'''
    # For now, extract the cases where model was > 95% sure
    en_articles = list(df_prob[df_prob['en_prob']>=prob]['fname'])
    ru_articles = list(df_prob[df_prob['ru_prob']>=prob]['fname'])
    return en_articles,ru_articles


def prepare_output(lang_code,article_list):
    '''Prepare the JSON output in the desired manner
    TODO : Make sure lang_code is a valid ISO 639-1 
    two-letter language code'''
    
    d = {"lang_code" : lang_code,"articles":article_list}
    return d


def languages(path,**kwargs):
    '''This function outputs the EN and RU 
    articles in the provided path. 
    If full path of the files are required,
    pass full_path = True'''
    file_list = read_filelist(path)
    logger.info(f'Number of files : {len(file_list)}')
    
    with Pool() as pool:
        results = pool.map(detect_distributed, file_list)
    
    df_prob = pd.DataFrame(results)
    df_prob['html_dict'] = list(results)
    
    try:
        full_path = kwargs['full_path']
        if full_path:
            df_prob['fname'] = [f for f in file_list]
        else:
            df_prob['fname'] = [os.path.basename(f) for f in file_list]
    except KeyError as e:
        logger.warning('Returning only the file name')
        df_prob['fname'] = [os.path.basename(f) for f in file_list]
    

    # For now, extract the cases where model was > 95% sure
    try:
        threshold = kwargs['threshold']
    except KeyError as e:
        logger.info("Setting default probabiltiy at 0.95")
        threshold = 0.95
    
    en_articles,ru_articles = label_final_lang(df_prob,prob=threshold)
    
    output = prepare_output("en",en_articles),prepare_output("ru",ru_articles)
    
    try:
        return_parsed_df = kwargs['return_parsed_df']
        if return_parsed_df:
            return output,df_prob
        else:
            return output
    except KeyError as e:
        logger.info('Returning output dictionary only.')
        return output

if __name__ == "__main__":
    if(len(sys.argv)) > 1:
        path = sys.argv[1]
        logger.info('SOURCE_DIR : '+ path)
        print(languages(path))