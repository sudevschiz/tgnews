import sys
import task1_languages as t1
import task2_news as t2
import task3_categories as t3
import task4_threads as t4
import task5_top as t5
import json

import logging

from time import time


### LOGGER
logFormatter = '%(asctime)s - %(levelname)s - %(message)s'
logging.basicConfig(format=logFormatter, level=logging.DEBUG)
logger = logging.getLogger(__name__)



if __name__ == "__main__":
    # Time the call
    st_time = time()
    
    if sys.argv[1] == "languages":
        j_out = t1.languages(sys.argv[2])
#         print(j_out)
    elif sys.argv[1] == "news":
        logger.info("Finding news articles in folder: "+sys.argv[2])
        j_out,_ = t2.news(path = sys.argv[2])
#         print(j_out)
    elif sys.argv[1] == "categories":
        logger.info("Tagging categories in folder: "+sys.argv[2])
        j_out,_= t3.categories(path = sys.argv[2])
#         print(j_out)
    elif sys.argv[1] == "threads":
        logger.info("Finding threads in folder: "+sys.argv[2])
        j_out,_ = t4.threads(sys.argv[2])
#         print(j_out)
    elif sys.argv[1] == "top":
        logger.info("Threads sorted by relevance in folder: "+sys.argv[2])
        j_out = t5.top(sys.argv[2])
#         print(j_out)
    else:
        logger.error("Unsupported arguments")
        exit()

    print(json.dumps(j_out, indent=4, sort_keys=True))
    
    logger.info(f"Execution complete in {time()-st_time} seconds.")