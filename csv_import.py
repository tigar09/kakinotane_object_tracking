import csv
import pandas as pd


def import_csv(csv_file_path):
    output_dict = {}
    with open(csv_file_path, 'r') as csv_file:
        csv_reader = csv.reader(csv_file)
        next(csv_reader)  # ヘッダー行をスキップする
        for row in csv_reader:
            if len(row) == 4:
                model_name, file_name, model_layers, model_parameters = row
                output_dict[model_name] = file_name, model_layers, model_parameters

    df = pd.DataFrame.from_dict(output_dict, orient='index', columns=['file_name', 'layers', 'parameters'])
    df = df.drop('file_name', axis=1)
    return output_dict, df

