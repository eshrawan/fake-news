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

#sklearn is the main ML package used in this program for modelling and accuracy measures

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

with open(os.path.join(basepath, 'train_val_data.pkl'), 'rb') as f:
  train_data, val_data = pickle.load(f)

#for testing downloaded data
print('Number of train examples:', len(train_data))
print('Number of val examples:', len(val_data))

#this class is to decide and judge individual factors. my approach is to test
#each method individually and work to aggregate it? my research shows the following
#methods are popular for NLP Analysis

#this gets the description from websites, beecause most fake news websites may not
#have a descriptor or may be gramatically incorrect
def get_description_from_html(html):
  soup = bs(html)
  description_tag = soup.find('meta', attrs={'name':'og:description'}) or soup.find('meta', attrs={'property':'description'}) or soup.find('meta', attrs={'name':'description'})
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

print('Description of Google.com:')
print(scrape_description('google.com'))

#implementation of a bag of words model?

#implementation on enitre html text: too much time.

def get_descriptions_from_data(data):
  # A dictionary mapping from url to description for the websites in
  # train_data.
  descriptions = []
  for site in tqdm(data):
    url, html, label = site
    descriptions.append(get_description_from_html(html))
  return descriptions

train_descriptions = get_descriptions_from_data(train_data)
train_urls = [url for (url, html, label) in train_data]

val_descriptions = get_descriptions_from_data(val_data)
print('\nNYTimes Description:')
print(train_descriptions[train_urls.index('nytimes.com')])

#counts the most frequqent words used in website descriptions.

#trial and error
vectorizer = CountVectorizer(max_features=300)

vectorizer.fit(train_descriptions)

def vectorize_data_descriptions(descriptions, vectorizer):
  X = vectorizer.transform(descriptions).todense() #transforms data to correct dimensions
  return X

print('\nPreparing train data...')
train_X = vectorize_data_descriptions(train_descriptions, vectorizer)
train_y = [label for url, html, label in train_data]

print('\nPreparing val data...')
val_X = vectorize_data_descriptions(val_descriptions, vectorizer)
val_y = [label for url, html, label in val_data]

model = LogisticRegression()

model.fit(train_X, train_y) #models on the basis of train_x, train_y
train_y_pred = model.predict(train_X)
print('Bag of words accuracy measures:\n')
print('Train accuracy', accuracy_score(train_y, train_y_pred))

val_y_pred = model.predict(val_X)
print('Val accuracy', accuracy_score(val_y, val_y_pred))

prf = precision_recall_fscore_support(val_y, val_y_pred)


print('Precision:', prf[0][1])
print('Recall:', prf[1][1])
print('F-Score:', prf[2][1])

VEC_SIZE = 300
glove = GloVe(name='6B', dim=VEC_SIZE)

# Returns word vector for word if it exists, else return None.
def get_word_vector(word):
    try:
      return glove.vectors[glove.stoi[word.lower()]].numpy()
    except KeyError:
      return None

#assuming the vectors are formed, their similarity can be shown by their parallel components, a cosine similarity

def cosine_similarity(vec1, vec2):
  return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

#word1 = "good"
#word2 = "better"

#print('Word 1:', word1)
#print('Word 2:', word2)

def cosine_similarity_of_words(word1, word2):
  vec1 = get_word_vector(word1)
  vec2 = get_word_vector(word2)

  if vec1 is None:
    print(word1, 'is not a valid word. Try another.')
  if vec2 is None:
    print(word2, 'is not a valid word. Try another.')
  if vec1 is None or vec2 is None:
    return None

  return cosine_similarity(vec1, vec2)

#transforms all words in a description to a vector form and adds it to X[i]
def glove_transform_data_descriptions(descriptions):
    X = np.zeros((len(descriptions), VEC_SIZE))
    for i, description in enumerate(descriptions):
        found_words = 0.0
        description = description.strip()
        for word in description.split():
            vec = get_word_vector(word)
            if vec is not None:
                # Increment found_words and add vec to X[i].
                X[i] += vec
                found_words += 1
        # divide the sum by the number of words added, so there is the
        # average word vector.
        if found_words > 0:
            X[i] /= found_words

    return X

glove_train_X = glove_transform_data_descriptions(train_descriptions)
glove_train_y = [label for (url, html, label) in train_data]

glove_val_X = glove_transform_data_descriptions(val_descriptions)
glove_val_y = [label for (url, html, label) in val_data]

model = LogisticRegression()
model.fit(glove_train_X, glove_train_y)
train_y_pred = model.predict(glove_train_X)
print('Glove Vector Accuracy:\n')
print('Train accuracy', accuracy_score(glove_train_y, train_y_pred))

val_y_pred = model.predict(glove_val_X)
print('Val accuracy', accuracy_score(glove_val_y, val_y_pred))

prf = precision_recall_fscore_support(glove_val_y, val_y_pred)

print('Precision:', prf[0][1])
print('Recall:', prf[1][1])
print('F-Score:', prf[2][1])

def train_model(train_X, train_y, val_X, val_y):
  model = LogisticRegression(solver='liblinear')
  model.fit(train_X, train_y)

  return model


def train_and_evaluate_model(train_X, train_y, val_X, val_y):
  model = train_model(train_X, train_y, val_X, val_y)
  train_y_pred = model.predict(train_X)
  print('Train accuracy', accuracy_score(train_y, train_y_pred))

  val_y_pred = model.predict(val_X)
  print('Val accuracy', accuracy_score(val_y, val_y_pred))

  prf = precision_recall_fscore_support(val_y, val_y_pred)

  print('Precision:', prf[0][1])
  print('Recall:', prf[1][1])
  print('F-Score:', prf[2][1])

  return model

#now the task is to test different domains of websites and detemrine whether it is fake from that
#the only approach is to brute-force in all possibilities

def prepare_data(data, featurizer):
    X = []
    y = []
    for datapoint in data:
        url, html, label = datapoint
        # We convert all text in HTML to lowercase, so <p>Hello.</p> is mapped to
        # <p>hello</p>. This will help us later when we extract features from
        # the HTML, as we will be able to rely on the HTML being lowercase.
        html = html.lower()
        y.append(label)

        features = featurizer(url, html)

        # Gets the keys of the dictionary as descriptions, gets the values
        # as the numerical features.
        feature_descriptions, feature_values = zip(*features.items())

        X.append(feature_values)

    return X, y, feature_descriptions

# Gets the log count of a phrase/keyword in HTML (transforming the phrase/keyword
# to lowercase).
def get_normalized_count(html, phrase):
    return math.log(1 + html.count(phrase.lower()))

# Returns a dictionary mapping from plaintext feature descriptions to numerical
# features for a (url, html) pair.
def keyword_featurizer(url, html):
    features = {}
    features['.com domain'] = url.endswith('.com')
    features['.org domain'] = url.endswith('.org')
    features['.net domain'] = url.endswith('.net')
    features['.info domain'] = url.endswith('.info')
    features['.org domain'] = url.endswith('.org')
    features['.biz domain'] = url.endswith('.biz')
    features['.ru domain'] = url.endswith('.ru')
    features['.co.uk domain'] = url.endswith('.co.uk')
    features['.co domain'] = url.endswith('.co')
    features['.tv domain'] = url.endswith('.tv')
    features['.news domain'] = url.endswith('.news')
    features['.edu domain'] = url.endswith('.edu')
    features['.int domain'] = url.endswith('.int')
    features['.gov domain'] = url.endswith('.gov')
    features['.mil domain'] = url.endswith('.mil')
    features['.app domain'] = url.endswith('.app')
    features['.wiki domain'] = url.endswith('.wiki')
    features['.www'] = url.startswith('www.')

#keywords that would indicate both real and fake news sources
    keywords = ['truth', 'apparently', 'fact', '!!!', 'proven', 'alleged', 'sources', 'references', 'opinion', '<img', 'editor', 'write to', 'datetime']

    for keyword in keywords:
      features[keyword + ' keyword'] = get_normalized_count(html, keyword)

    return features

train_X, train_y, feature_descriptions = prepare_data(train_data, keyword_featurizer)
val_X, val_y, feature_descriptions = prepare_data(val_data, keyword_featurizer)

print('Keyword Feautrizer accuracy:\n')
model = train_and_evaluate_model(train_X, train_y, val_X, val_y)

vectorizer = CountVectorizer(max_features=7)

vectorizer.fit(train_descriptions)

def vectorize_data_descriptions(data_descriptions, vectorizer):
  X = vectorizer.transform(data_descriptions).todense()
  return X

bow_train_X = vectorize_data_descriptions(train_descriptions, vectorizer)
bow_train_y = [label for url, html, label in train_data]

bow_val_X = vectorize_data_descriptions(val_descriptions, vectorizer)
bow_val_y = [label for url, html, label in val_data]

model = train_and_evaluate_model(bow_train_X, bow_train_y, bow_val_X, bow_val_y)

VEC_SIZE = 300
glove = GloVe(name='6B', dim=VEC_SIZE)

# Returns word vector for word if it exists, else return None.
def get_word_vector(word):
    try:
      return glove.vectors[glove.stoi[word.lower()]].numpy()
    except KeyError:
      return None

def combine_features(X_list):
  return np.concatenate(X_list, axis=1)

#combining all three features together
X_list = [train_X, bow_train_X, glove_train_X]
val_list = [val_X, bow_val_X, glove_val_X]

combined_train_X = combine_features(X_list)
combined_val_X = combine_features(val_list)
print("Combined model accuracy\n")
model = train_and_evaluate_model(combined_train_X, train_y, combined_val_X, val_y)

#the previous methods were only for testing individually, below all these methods are combined
def get_data_pair(url):
  if not url.startswith('http'):
      url = 'http://' + url
  url_pretty = url
  if url_pretty.startswith('http://'):
      url_pretty = url_pretty[7:]
  if url_pretty.startswith('https://'):
      url_pretty = url_pretty[8:]

  # Scrape website for HTML
  response = requests.get(url, timeout=10)
  htmltext = response.text

  return url_pretty, htmltext


#enter website link here
curr_url = "https://local.theonion.com/school-shooter-thankfully-stopped-before-doing-enough-d-1838201037"
url, html = get_data_pair(curr_url) #since the other functions work with the html and url separately


def dict_to_features(features_dict):
  X = np.array(list(features_dict.values())).astype('float')
  X = X[np.newaxis, :] #just to make sure the dimensions are correct
  return X
def featurize_data_pair(url, html):
  keyword_X = dict_to_features(keyword_featurizer(url, html))
  description = get_description_from_html(html)
  bow_X = vectorize_data_descriptions([description], vectorizer)
  glove_X = glove_transform_data_descriptions([description])

#combine the glove vector, the bow transform and the keyword featurizer)

  X = combine_features([keyword_X, bow_X, glove_X, ])

  return X

curr_X = featurize_data_pair(url, html)

model = train_model(combined_train_X, train_y, combined_val_X, val_y)

curr_y = model.predict(curr_X)[0]


if curr_y < .5: #may not be the best threshold. but brings the highest accuracy for keywords chosen
  print(curr_url, 'appears to be real.')
else:
  print(curr_url, 'appears to be fake.')
