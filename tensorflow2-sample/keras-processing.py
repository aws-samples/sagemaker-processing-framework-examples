
import os
import argparse
import tarfile
import numpy as np
from tensorflow.keras.models import load_model

def model_loading_logic(model_dir):
    # How do you load the model?
    print(f'Loading model from {model_dir} .')
    model = load_model(model_dir)
    print('Model loaded successfully.')
    return model

def processing_logic(data, model):
    # How do you process your data? In this case, we predict
    return model.predict(data)

# Parse the argument
parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, default='/opt/ml/processing/input/data')
parser.add_argument('--model_path', type=str, default='/opt/ml/processing/input/model')
parser.add_argument('--output_path', type=str, default='/opt/ml/processing/output')
args = parser.parse_args()

# extract all data from the model.tar.gz file
model_path = '/tmp/model/'
os.mkdir(model_path)
with tarfile.open(os.path.join(args.model_path, 'model.tar.gz'), 'r:gz') as m:
    def is_within_directory(directory, target):
        
        abs_directory = os.path.abspath(directory)
        abs_target = os.path.abspath(target)
    
        prefix = os.path.commonprefix([abs_directory, abs_target])
        
        return prefix == abs_directory
    
    def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
    
        for member in tar.getmembers():
            member_path = os.path.join(path, member.name)
            if not is_within_directory(path, member_path):
                raise Exception("Attempted Path Traversal in Tar File")
    
        tar.extractall(path, members, numeric_owner=numeric_owner) 
        
    
    safe_extract(m, model_path)
print('Files extracted:')
print(os.listdir(model_path))
        
# Load the model
model = model_loading_logic(model_path)

# Read the input files from data_path and generate an output
output = []
files = os.listdir(args.data_path)
for f in files:
    data = np.load(os.path.join(args.data_path, f))
    output.append(processing_logic(data, model))

# Save the predictions locally, SageMaker will store to S3
with open(os.path.join(args.output_path, 'output.txt'), 'w') as w:
    for o in output[0]:
        w.write(str(o)+'\n')
print(f"Output wrote correctly to {os.path.join(args.output_path, 'output.txt')}")
