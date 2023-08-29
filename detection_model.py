from ultralytics.models import YOLO,RTDETR

import csv_import

PATH = './model'
dict_model, df = csv_import.import_csv(PATH + '/model.csv')

def df_set():
    return df

def set_st_radio():
    set_st_radio = [model for model in dict_model]
    return set_st_radio


def select_model(radio_model):
    set_model = PATH + '/' + dict_model[radio_model][0]

    #自作データーセットを利用して学習したデータ
    if 'DETR' in radio_model:
        model = RTDETR(set_model)
    else:
        model = YOLO(set_model)
    return model