# Récupération de l'objet Python avec joblib

import os
from joblib import load

# Import the `numpy` library as `np`
import numpy as np

import json
def init_db():
    folder = './joblib_memmap'
    data_filename_memmap = os.path.join(folder, 'data_memmap')
    data = load(data_filename_memmap, mmap_mode='r')
    return data

def format_rep(data, recommendation):
    tab_res = []

    my_dict = {'_request':{"id":recommendation['movie_id'].values[0], "name": recommendation['Film'].values[0]}}

    for i in range(1,6):
        new_id = recommendation['R{}'.format(i)].values[0]
        new_title = data.Film[np.where(data["movie_id"] == new_id)[0]].values[0]
        new_dict= {"id":new_id, "name": new_title}
        tab_res.append(new_dict)
    my_dict['_results'] = tab_res
    return my_dict

def recom_movie(movie_id):
    movie_base = init_db()
    # Consultation des recommendations du film
    try:
        index_movie = np.where(movie_base["movie_id"] == movie_id)[0]
        resultat = movie_base.iloc[index_movie]
        # Formatage du résultat
        res_dict = format_rep(movie_base, resultat)
    except:
        res_dict = """Erreur - veuillez vérifier que votre url est de la forme
                    https://oc-jalvarez-p3.herokuapp.com/recommend/{ID_FILM},
                    où {ID_FILM} figure dans la base de films"""
    return res_dict
