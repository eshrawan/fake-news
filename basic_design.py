import math
import urllib as urllib2
import os
import numpy as np
from bs4 import BeautifulSoup as bs
import requests
from tqdm import tqdm
from sklearn.feature_extraction.text import CountVectorizer
from torchtext.vocab import GloVe

import pickle

import requests, io, zipfile
r = requests.get("https://www.dropbox.com/s/2pj07qip0ei09xt/inspirit_fake_news_resources.zip?dl=1")
z = zipfile.ZipFile(io.BytesIO(r.content))
z.extractall()

basepath = '.'

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

with open(os.path.join(basepath, 'train_val_data.pkl'), 'rb') as f:
  train_data, val_data = pickle.load(f)

print('Number of train examples:', len(train_data))
print('Number of val examples:', len(val_data))
