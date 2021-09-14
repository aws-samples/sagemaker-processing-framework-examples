import argparse
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from gluonts.dataset import common

parser = argparse.ArgumentParser()
parser.add_argument('--input_path', type=str, default='/opt/ml/processing/input/data/')
parser.add_argument('--output_path', type=str, default='/opt/ml/processing/output/')
args = parser.parse_args()

files = os.listdir(args.input_path)
for f in files:
    df = pd.read_csv(os.path.join(args.input_path, f))
    train, test = train_test_split(df, test_size=0.25, shuffle=False)
    train_data = common.ListDataset([{
        "start": df.index[0],
        "target": df.value[:"2015-04-05 00:00:00"]
    }], freq="5min")
    test_data = common.ListDataset([{
        "start": df.index[0],
        "target": df.value["2015-04-05 00:00:00":]
    }], freq="5min")
    metadata = common.MetaData(
        freq="5min", target={'name':'value'}, 
        prediction_length=12
    )
    common.TrainDatasets(metadata, train_data, test_data).save(args.output_path)
    print(f'Generated dataset for file {f}')