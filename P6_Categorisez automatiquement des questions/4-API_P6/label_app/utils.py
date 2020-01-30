
import pandas as pd
import numpy as np
import os
from joblib import load
from bs4 import BeautifulSoup
from nltk import download
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
download('stopwords')
download('wordnet')
from flask import flash

def init_obj():
    """
    Récupération des objets Python avec joblib
    """
    # répertoire joblib
    folder = './joblib_memmap'
    # Objets
    data_filename_memmap = os.path.join(folder, 'worddict_memmap')
    wordDict = load(data_filename_memmap, mmap_mode='r')
    data_filename_memmap = os.path.join(folder, 'vector_memmap')
    vectorizer = load(data_filename_memmap, mmap_mode='r')
    data_filename_memmap = os.path.join(folder, 'tokeniz_memmap')
    tokenizer = load(data_filename_memmap, mmap_mode='r')
    data_filename_memmap = os.path.join(folder, 'tfidf_memmap')
    tfidf_transformer = load(data_filename_memmap, mmap_mode='r')
    data_filename_memmap = os.path.join(folder, 'svc_memmap')
    gs_svc = load(data_filename_memmap, mmap_mode='r')
    data_filename_memmap = os.path.join(folder, 'labels_memmap')
    label_names = load(data_filename_memmap, mmap_mode='r')

    return wordDict, vectorizer, tokenizer, tfidf_transformer, gs_svc, label_names

def word_replace(text, wd):
    """
    Replace words found in Worddict
    """
    for key in wd:
        text = text.replace(key, wd[key])
    return text


def tokenize_body(body_full, tokenizer):
    """
    Tokenisation du post avec lemmatisation et suppression des stopwords
    """
    wordnet_lemmatizer = WordNetLemmatizer()
    sw = stopwords.words('english')
    list_a = tokenizer.tokenize(body_full)
    token_list = [wordnet_lemmatizer.lemmatize(word) for word in list_a if (word not in sw and not word.isdigit())]
    return " ".join(token_list)

def preprocess_api(X, wd, tokenizer, vectorizer):
    """
    Préprocessing des posts
    """
    X['Body'] = X['Title'] + ' ' + X['Body']
    X['Body'] = X['Body'].apply(lambda x: BeautifulSoup(x, 'html.parser').text)
    X['Body'] = X['Body'].apply(lambda x: x.lower())
    X['Body'] = X['Body'].apply(lambda x: word_replace(x, wd))
    X['Body'] = X['Body'].apply(lambda x: tokenize_body(x, tokenizer))

    X_count = vectorizer.transform(X['Body'])
    X_preprocessed = X_count.toarray()
    return X_preprocessed

def label_pred(form):
    """
    Prédiction des labels
    """
    # On récupère les champs renseignés
    title = form.title_raw.data
    body = form.body_raw.data
    X = pd.DataFrame(np.array([[title, body]]), columns = ['Title', 'Body'])
    # Chargement des objets Python
    wordDict, vectorizer, tokenizer, tfidf_transformer, gs_svc, label_names = init_obj()
    # Preprocessing
    X_preproc = preprocess_api(X, wordDict, tokenizer, vectorizer)
    X_tfidf = tfidf_transformer.transform(X_preproc).toarray()
    # Prédiction
    y_pred = []
    try:
        y_pred_idx = gs_svc.predict(X_tfidf)[0]
        # y_pred = y_pred_idx
        for index  in range(len(y_pred_idx)):
          if y_pred_idx[index] == 1:
            y_pred.append(label_names[index])
    except:
        y_pred = None
    return y_pred
