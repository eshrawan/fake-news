import math
import urllib as urllib2
import os
import numpy as np
from bs4 import BeautifulSoup as bs
import requests
from tqdm import tqdm
from sklearn.feature_extraction.text import CountVectorizer
from torchtext.vocab import GloVe #to run this code, one will need to install pip packages for
#each of these functions

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

#for testing downloaded data
#print('Number of train examples:', len(train_data))
#print('Number of val examples:', len(val_data))

#basically right now i'm deciding what factors to select. my approach is to test
#each method individually and work to aggregate it? my research shows the following
#methods are popular for NLP Analysis

#this gets the description from websites, beecause most fake news websites may not
#have a descriptor or may be gramatically incorrect
def get_description_from_html(html):
  soup = bs(html)
  description_tag = soup.find('meta', attrs={'name':'og:description'}) or soup.find('meta', attrs={'property':'description'}) or
  soup.find('meta', attrs={'name':'description'})
  if description_tag:
    description = description_tag.get('content') or ''
  else: # If there is no description, return empty string.
    description = ''
  return description

def scrape_description(url):
  if not url.startswith('http'):
    url = 'http://' + url #bs works with what it calls valid links, so add http
  response = requests.get(url, timeout=10)
  html = response.text
  description = get_description_from_html(html)
  return description

#print('Description of Google.com:')
#print(scrape_description('google.com'))

#implementation of a bag of words model?
