from Mylib import myfuncs
import time
import os
import re
from sklearn.pipeline import Pipeline
from src.utils import classes
import pandas as pd
import plotly.express as px


def get_batch_size_from_model_training_name(name):
    pattern = r"batch_(\d+)"
    return int(re.findall(pattern, name)[0])
