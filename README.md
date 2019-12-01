# TELEGRAM DATA CLUSTERING CONTEST


## Overview

This repo is our attemp on the Telegram Data Clustering contest.
More details here : https://contest.com/docs/data_clustering


## Setting up:
1. The submission folder contains the files that will be finally shared for submission.
2. Install the dependencies in `deb_packages.txt`
3. `site_packages` are prepared to have the correct dependencies for the debian system that will be used for the evaluation as mentioned in the contest page.
4. You can create a `virtualenv` and install the pythin dependencies listed in `requirements.txt` by doing `pip install -r requirements.txt`
5. Additionally, some pre-trained vectors need to the downloaded and added to the assets folders.

## Execution

1. `tgnews` executable has been set up specifically for the contest.
2. For testing, run the modules using 
    ```
    (venv) submission :  python src/main.py languages 'test/'
    
    ```
    and so on.
    
    
## Explorations

1. Jupyter notebooks show the different EDA, trials and modelling section.
