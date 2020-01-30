
import pandas as pd
import numpy as np
import os
from joblib import load
from sklearn import preprocessing, linear_model
import json
import datetime
from flask import flash

def init_obj():
    """
        Récupération des objets Python avec joblib
    """
    # répertoire joblib
    folder = './joblib_memmap'
    # Régression
    data_filename_memmap = os.path.join(folder, 'reg_memmap')
    reg = load(data_filename_memmap, mmap_mode='r')

    # Eléments de feature
    data_filename_memmap = os.path.join(folder, 'feat1_memmap')
    days_cat = load(data_filename_memmap, mmap_mode='r')

    data_filename_memmap = os.path.join(folder, 'feat2_memmap')
    month_cat = load(data_filename_memmap, mmap_mode='r')

    data_filename_memmap = os.path.join(folder, 'feat3_memmap')
    colonnes = load(data_filename_memmap, mmap_mode='r')
    # X_feat_3 = pd.DataFrame(columns = colonnes)

    data_filename_memmap = os.path.join(folder, 'feat4_memmap')
    dep_cat = load(data_filename_memmap, mmap_mode='r')

    data_filename_memmap = os.path.join(folder, 'feat5_memmap')
    arr_cat = load(data_filename_memmap, mmap_mode='r')

    data_filename_memmap = os.path.join(folder, 'feat6_memmap')
    time_cat = load(data_filename_memmap, mmap_mode='r')

    # Standardisation
    data_filename_memmap = os.path.join(folder, 'std_memmap')
    std_scale = load(data_filename_memmap, mmap_mode='r')

    return reg, days_cat, month_cat, colonnes, dep_cat, arr_cat, std_scale, time_cat

def init_sel():
    """
    Chargement des listes utilisées dans le formulaire
    """
    # répertoire joblib
    folder = './joblib_memmap'
    # Liste des couples compagnie/ID
    data_filename_memmap = os.path.join(folder, 'iata_memmap')
    cies = load(data_filename_memmap, mmap_mode='r')

    # Liste des couples aéroport/ID
    data_filename_memmap = os.path.join(folder, 'apt_memmap')
    airports = load(data_filename_memmap, mmap_mode='r')

    return cies, airports

def init_holidays():
    # Déclaration de variables
    holidays= ['2019-01-01', '2019-01-21', '2019-02-18', '2019-05-27',
               '2019-07-04', '2019-09-02', '2019-10-14', '2019-11-11',
               '2019-11-28', '2019-12-25']

    holidayDates = [datetime.datetime.strptime(x, '%Y-%m-%d') for x in holidays]
    return holidayDates

def DaysToHoliday(year, month, day):
  """calculate number of days from input date to next holiday
  """
  holidayDates = init_holidays()
  # Create a DATE object we can use to calculate the time difference
  currDate = datetime.datetime(year,month,day)

  numDays = min([abs(currDate - x) for x in holidayDates]) # Now find the minimum difference between the date and our holidays
  return(numDays.days)                                          # We can vectorize this to automatically find the minimum closest
                                                           # holiday by subtracting all holidays at once


def format_feat(form, days_cat, month_cat, colonnes, dep_cat, arr_cat, std_scale, time_cat):
    """
    Formattage des features
    """
    # Création du dataset avec les données issues du formulaire
    # datee = datetime.datetime.strptime(str(form.date_vol.data), "%Y-%m-%d")
    datee = datetime.datetime.strptime(str(form.date_vol.data), "%Y-%m-%d")
    heure = datetime.datetime.strptime(str(form.heure_vol.data), "%H:%M:%S")
    d = {
            'DAY_OF_WEEK': [datee.isoweekday()],
            'YEAR': [datee.year],
            'MONTH': [datee.month],
            'DAY_OF_MONTH': [datee.day],
            'HOUR': [heure.hour],
            'AIRLINE_ID': [int(form.cie_id.data)],
            'ORIGIN_AIRPORT_ID': [int(form.dep_id.data)],
            'DEST_AIRPORT_ID': [int(form.arr_id.data)]
        }
    X_form = pd.DataFrame(data=d)

    # Mise en forme des features
    X_pred_1 = X_form['DAY_OF_WEEK'].map(days_cat)
    X_pred_2 = X_form['MONTH'].map(month_cat)

    # X_pred_3 = pd.DataFrame(0, index = [0], columns = X_feat_3.columns)
    X_pred_3 = pd.DataFrame(0, index = [0], columns = colonnes)

    id_cie = X_form['AIRLINE_ID'].values[0]
    X_pred_3[id_cie][0] = 1

    X_pred_4 = X_form['ORIGIN_AIRPORT_ID'].map(dep_cat)
    X_pred_5 = X_form['DEST_AIRPORT_ID'].map(arr_cat)

    hour_blk = {0: '0001-0559',
            1: '0001-0559',
            2: '0001-0559',
            3: '0001-0559',
            4: '0001-0559',
            5: '0001-0559',
            6: '0600-0659',
            7: '0700-0759',
            8: '0800-0859',
            9: '0900-0959',
            10: '1000-1059',
            11: '1100-1159',
            12: '1200-1259',
            13: '1300-1359',
            14: '1400-1459',
            15: '1500-1559',
            16: '1600-1659',
            17: '1700-1759',
            18: '1800-1859',
            19: '1900-1959',
            20: '2000-2059',
            21: '2100-2159',
            22: '2200-2259',
            23: '2300-2359'
           }
    X_form['DEP_TIME_BLK'] = X_form['HOUR'].map(hour_blk)

    X_pred_6 = X_form['DEP_TIME_BLK'].map(time_cat)

    X_pred_7 = pd.DataFrame(0, index = [0], columns = ['HDAYS'])
    X_pred_7['HDAYS'][0] = DaysToHoliday(X_form['YEAR'],X_form['MONTH'], \
                                         X_form['DAY_OF_MONTH'])

    X_pred = pd.concat([X_pred_1, X_pred_2, X_pred_3, X_pred_4, X_pred_5, \
                        X_pred_6, X_pred_7], axis = 1, join = 'inner')
    X_pred.fillna(0, inplace=True)

    # Standardisation des features
    X_pred_std = std_scale.transform(X_pred)

    return X_pred_std


def delay_pred(form):
    """
    Prédiction du retard
    """
    # Chargement des objets Python
    reg, days_cat, month_cat, col_cies, dep_cat, arr_cat, std_scale, time_cat = init_obj()
    # Formatage des features
    X_pred_std = format_feat(form, days_cat, month_cat, col_cies, dep_cat,\
                             arr_cat, std_scale, time_cat)
    # Prédiction
    try:
        y_pred = int(round(reg.predict(X_pred_std)[0], 0))
    except:
        y_pred = None
    return y_pred
