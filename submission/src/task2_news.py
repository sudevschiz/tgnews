
import lightgbm as lgb

from multiprocessing import Pool

from helper_functions import *
import task1_languages as t1


### ASSET LOCATIONS
ru_vec = ""
en_vec = "../assets/wiki-news-300d-1M.vec"

model_file = "../assets/models/lgb_news_predict.txt"


### NECESSARY FUNCTIONS
def prepare_output(y_pred,file_list,threshold=0.5,**kwargs):
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
  
    # Call the languages module to get only EN and RU articles
    # Enable full_path parameter for later manipulations
    lang_json,df_parsed = t1.languages(sys.argv[1],full_path=True,return_parsed_df=True)
    
    file_list = []
    for lang in lang_json:
        file_list.extend(lang['articles'])
    
    # Subset the data frame to only consider EN and RU
    df_parsed = df_parsed[df_parsed.fname.isin(file_list)]
    # Load the predictive model that classifies news and no news
    model = lgb.Booster(model_file=model_file)
    
    
    #### START EXECUTION ON FILES

    text_list = list(df_parsed['all_text'])
    
    # Create features from the text
    # The below parallel operation doesn't work
   
    # with Pool(N_CORES) as pool:
    #    vector = pool.map(compute_ft_sum,text_list)
    
    _,_,ft_dict = load_vectors(en_vec)
    vector = [compute_ft_sum(t,ft_dict=ft_dict) for t in text_list]
    
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

