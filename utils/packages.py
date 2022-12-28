import numpy as np
import pandas as pd

import os, nltk, string, gensim, warnings, pickle

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize 

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from sklearn import preprocessing, ensemble, metrics
from sklearn.metrics import classification_report

import matplotlib.pyplot as plt