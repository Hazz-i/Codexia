# import library
import string
import pickle
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from utils import JSONParser


# load data
path = "dataset/data.json"
jp = JSONParser()
jp.parse(path)
df = jp.get_dataframe()