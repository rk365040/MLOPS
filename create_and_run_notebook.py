import boto3
import json
import nbformat
import time

# SageMaker configurations
sagemaker_client = boto3.client("sagemaker")
sagemaker_runtime = boto3.client("sagemaker-runtime")
s3_client = boto3.client("s3")

domain_id = "d-0ufq0tljpepd"
user_profile_name = "MLOPS"
notebook_name = "mlops_activity3.ipynb"

# Code to be inserted into the notebook
notebook_code = """\
# Use the previously prepared data
from sagemaker import Session
import sagemaker
import boto3
import numpy as np
import pandas as pd
import os
from sagemaker import get_execution_role

role = get_execution_role()
bucket = sagemaker.Session().default_bucket()
prefix = 'mlops/activity-3'
sess = Session()

train_path = f"s3://{bucket}/{prefix}/train"
validation_path = f"s3://{bucket}/{prefix}/validation"
test_path = f"s3://{bucket}/{prefix}/test"

container = sagemaker.image_uris.retrieve(region=boto3.Session().region_name, framework='xgboost', version='latest')
s3_input_train = sagemaker.inputs.TrainingInput(s3_data=f's3://{bucket}/{prefix}/train', content_type='csv')
s3_input_validation = sagemaker.inputs.TrainingInput(s3_data=f's3://{bucket}/{prefix}/validation/', content_type='csv')

xgb = sagemaker.estimator.Estimator(container, role, instance_count=1, instance_type='ml.m4.xlarge',
                                    output_path=f's3://{bucket}/{prefix}/output', sagemaker_session=sess)

xgb.set_hyperparameters(max_depth=5, eta=0.2, gamma=4, min_child_weight=6, subsample=0.8, silent=0, objective='binary:logistic', num_round=100)

xgb.fit({'train': s3_input_train, 'validation': s3_input_validation}) 

xgb_predictor = xgb.deploy(initial_instance_count=1, instance_type='ml.m4.xlarge')

xgb_predictor.serializer = sagemaker.serializers.CSVSerializer()

test_data_x = pd.read_csv(os.path.join(test_path, 'test_script_x.csv'), header=None)
test_data_y = pd.read_csv(os.path.join(test_path, 'test_script_y.csv'), header=None)

def predict(data, predictor, rows=500):
    split_array = np.array_split(data, int(data.shape[0] / float(rows) + 1))
    predictions = ''
    for array in split_array:
        predictions = ','.join([predictions, predictor.predict(array).decode('utf-8')])
    return np.fromstring(predictions[1:], sep=',')

predictions = predict(test_data_x, xgb_predictor)
pd.crosstab(index=test_data_y[0], columns=np.round(predictions), rownames=['actuals'], colnames=['predictions'])
"""

# Convert the script into a Jupyter notebook
nb = nbformat.v4.new_notebook()
nb.cells.append(nbformat.v4.new_code_cell(notebook_code))

# Save the notebook as a .ipynb file
notebook_file = f"/tmp/{notebook_name}"
with open(notebook_file, "w") as f:
    nbformat.write(nb, f)

# Upload notebook to S3
s3_bucket = "<YOUR_S3_BUCKET>"
s3_key = f"sagemaker-notebooks/{notebook_name}"
s3_client.upload_file(notebook_file, s3_bucket, s3_key)
print(f"Notebook uploaded to s3://{s3_bucket}/{s3_key}")

# Start a notebook execution in SageMaker Studio
response = sagemaker_client.create_processing_job(
    ProcessingJobName="SageMakerNotebookExecution",
    ProcessingInputs=[
        {
            "InputName": "source-notebook",
            "S3Input": {
                "S3Uri": f"s3://{s3_bucket}/{s3_key}",
                "LocalPath": "/opt/ml/processing/input",
                "S3DataType": "S3Prefix",
                "S3InputMode": "File",
            },
        }
    ],
    ProcessingOutputConfig={
        "Outputs": [
            {
                "OutputName": "output-notebook",
                "S3Output": {
                    "S3Uri": f"s3://{s3_bucket}/sagemaker-notebooks-output/",
                    "LocalPath": "/opt/ml/processing/output",
                    "S3UploadMode": "EndOfJob",
                },
            }
        ]
    },
    ProcessingResources={
        "ClusterConfig": {
            "InstanceCount": 1,
            "InstanceType": "ml.t3.medium",
            "VolumeSizeInGB": 10,
        }
    },
    AppSpecification={
        "ImageUri": "763104351884.dkr.ecr.us-east-1.amazonaws.com/sagemaker-notebook-image",
        "ContainerEntrypoint": ["python3", "/opt/ml/processing/input/mlops_activity3.ipynb"],
    },
    RoleArn="<YOUR_SAGEMAKER_EXECUTION_ROLE>",
)

print("SageMaker notebook execution started:", response)
