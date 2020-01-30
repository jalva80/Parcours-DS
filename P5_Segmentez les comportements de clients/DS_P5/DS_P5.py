import sys
import os
import pandas as pd
import numpy as np
import datetime as dt

from joblib import load
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# initialisation de variables globales
#   chemin du fichier de description des notices
str_notice = './notice/categories.txt'
# répertoire joblib
joblib_folder = './joblib_memmap'

def init():
    """ Initialisation:
    Chargement du modèle entraîné pour classification .
    Renvoie le modèle chargé.
    """

    # modèle de Classification
    data_filename_memmap = os.path.join(joblib_folder, 'modele')
    clf = load(data_filename_memmap, mmap_mode='r')
    # standard scaler des données
    data_filename_memmap = os.path.join(joblib_folder, 'scaler')
    std_scaler = load(data_filename_memmap, mmap_mode='r')

    return clf, std_scaler

def load_file(str_file):
    """Fonction qui charge le fichier d'entrée.

    Keyword arguments:
    str_file -- chemin du fichier d'entrée

    Renvoie un dataset avec les données du fichier chargé si il existe.
    Renvoie une exception si le chemin est erroné.
    """
    raw_data = pd.read_excel(str_file)
    return raw_data

def preprocess(raw_data, std_scaler):
    """Effectue le preprocessing des données brutes
    Renvoie un dataset utilisable pour la prédiction
    """
    X1 = raw_data[raw_data.groupby("CustomerID")["InvoiceDate"].transform('min') == raw_data['InvoiceDate']]
    X2 = pd.DataFrame(X1.Quantity * X1.UnitPrice, columns = ['TotalPrice'])

    X = pd.concat([X1,X2],axis=1, join='outer')

    list_ID = sorted(X.CustomerID.unique())

    resultat = pd.DataFrame()

    for id in list_ID:
      df_ID = X[X['CustomerID'] == id]
      invoice_value = df_ID.TotalPrice.sum()
      qty = df_ID.Quantity.sum()
      resultat = resultat.append({'CustomerID': id,
                                  'monetary_value': invoice_value,     # Montant Total des achats
                                  'quantity': qty,   # qté de la cde
                                  'mean_price': invoice_value / qty,         # prix moyen par article
                                  'references': len(df_ID.StockCode.unique())     # nb de ref differentes
                                 }, ignore_index=True)

    resultat.set_index('CustomerID', inplace = True)
    resultat.sort_index(inplace = True)
    X_std = std_scaler.transform(resultat)
    return list_ID, X_std

def predict(features, clf):
    """ Fonction qui applique le modèle de prédiction aux features
    Renvoie la classe du client"""
    y_pred = clf.predict(features)

    return y_pred

def display_result(ID, categorie, str_notice):
    """ Fonction qui affiche la prédiction de classe du client et le contenu
    du fichier categories.txt qui contient la description des catégories"""
    for i in range(len(ID)):
        print("La catégorie du client {} est: {}".format(ID[i], int(categorie[i])))

    with open(str_notice) as file:
        data = file.read()
        print("\n", data)

if __name__ == "__main__":
    print("Analyse en cours. Veuillez patienter ...\n")
    str_file = sys.argv[1]
    classif, std_scaler = init()
    data = load_file(str_file)
    liste_ID, features = preprocess(data, std_scaler)
    customer_class = predict(features, classif)
    display_result(liste_ID, customer_class, str_notice)
