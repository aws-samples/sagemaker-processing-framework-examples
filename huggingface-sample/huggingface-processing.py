import argparse
import os
import pandas as pd
from transformers import pipeline
from sklearn.model_selection import train_test_split

# Argument parsing
parser = argparse.ArgumentParser()
parser.add_argument('--input_path', type=str, default='/opt/ml/processing/input/data/')
parser.add_argument('--output_path', type=str, default='/opt/ml/processing/output/')
args = parser.parse_args()

# Model loading
classifier = pipeline(
    "sentiment-analysis", 
    model='nlptown/bert-base-multilingual-uncased-sentiment'
)

# Data loading and processing
files = os.listdir(args.input_path)
df = pd.concat([pd.read_csv(os.path.join(args.input_path, f), sep='\t') for f in files], axis=0)
df['verified_purchase'] = df.verified_purchase.apply(lambda x: 1 if x=='Y' else 0)
df['sentiment'] = [int(prediction['label'][0]) for prediction in classifier(list(df.review_body.values))]
df = df.drop(['product_title', 'vine', 'review_headline', 'review_body', 'review_date'], axis=1)
df = df.drop(['review_id', 'product_id', 'customer_id', 'product_parent'], axis=1)
train, test = train_test_split(df, test_size=0.25)
test, val = train_test_split(test, test_size=0.4)

# Save locally to store to S3
train.to_csv(os.path.join(args.output_path, 'train/train.csv'), index=False)
test.to_csv(os.path.join(args.output_path, 'test/test.csv'), index=False)
val.to_csv(os.path.join(args.output_path, 'val/val.csv'), index=False, header=False)