import pandas as pd
import numpy as np
import random
import glob
import logging
import os
import re
from multiprocessing import Pool

from langdetect import detect_langs
import pycld2 as cld2