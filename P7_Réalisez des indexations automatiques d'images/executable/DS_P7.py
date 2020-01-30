import sys
import os
import warnings

import numpy as np

from joblib import load

with warnings.catch_warnings():
    warnings.simplefilter("ignore", ResourceWarning)
    # warnings.simplefilter("ignore")
    # import OpenCV library as cv2

    from keras.models import load_model
import cv2 as cv2
import h5py
import ctypes

# initialisation de variables globales
#   chemin du modèle entrainé
str_modele = './model/cnn_model.h5'
str_input = './input/'
str_target = './model/target'

def init(str_path):
    """ Initialisation:
    Chargement du modèle entraîné pour classification.

    Keyword arguments:
    str_path -- chemin du fichier d'entrée (modèle sauvagardé au format .h5)

    Renvoie le modèle chargé et la liste des target names.
    """
    # modèle de Classification
    cnn_model = load_model(str_path)
    # liste narget names
    target_names = load(str_target, mmap_mode='r')

    return cnn_model, target_names

def load_file(str_file):
    """Fonction qui charge le fichier d'entrée (image).

    Keyword arguments:
    str_file -- chemin du fichier d'entrée

    Renvoie un dataset avec les données du fichier chargé si il existe.
    Renvoie une exception si le chemin est erroné.
    """
    X = []
    img = cv2.imread(str_file)
    img = np.expand_dims(cv2.resize(img, (500,375)), axis=0)
    X.append(img)

    return X

def predict(features, clf):
    """ Fonction qui applique le modèle de prédiction aux features

    Keyword arguments:
    features --
    clf -- modele

    Renvoie la race prédite
    """
    y_pred = clf.predict(features)
    y_class = y_pred.argmax(axis=-1)
    return y_class[0]

def display_result(race):
    """ Fonction qui affiche la prédiction de race du chien

    Keyword arguments:
    race -- nom de la race à afficher

    Renvoie: None
    """
    msg = "La race du chien est: {}".format(race)
    print(msg)

    ctypes.windll.user32.MessageBoxW(0, msg, "Guess my dog!", 0)

if __name__ == "__main__":

    print("Analyse en cours. Veuillez patienter ...\n")
    # chemin complet vers l'image passée en argument
    str_file = str_input + sys.argv[1]
    # chargement du modèle
    cnn_model, target_names = init(str_modele)
    # chargement de l'image
    data = load_file(str_file)

    race_pred = target_names[predict(data, cnn_model)]

    display_result(race_pred)
