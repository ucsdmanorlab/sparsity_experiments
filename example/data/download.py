import boto3
import os
from botocore import UNSIGNED
from botocore.config import Config

# set bucket credentials
config=Config(signature_version=UNSIGNED)
bucket = "open-neurodata"

# connect to client
client = boto3.client('s3', config=config)
#client = boto3.client('s3', aws_access_key_id=access_key, aws_secret_access_key=secret_key)

# list data
client.list_objects(Bucket=bucket, Prefix="funke")

# download directory structure file - this shows exactly how the s3 data is stored
client.download_file(
    Bucket=bucket,
    Key="funke/structure.md",
    Filename="structure.md")

# function to download all files nested in a bucket path
def downloadDirectory(
    bucket_name,
    path):
  
    resource = boto3.resource(
        's3',
        config=config)
        #aws_access_key_id=access_key,
        #aws_secret_access_key=secret_key
        #)
    
    bucket = resource.Bucket(bucket_name)

    for obj in bucket.objects.filter(Prefix=path):
        if not os.path.exists(os.path.dirname(obj.key)):
            os.makedirs(os.path.dirname(obj.key))
        
        key = obj.key

        print(f'Downloading {key}')
        bucket.download_file(key, key)

# download example fib25 training data
downloadDirectory(
    bucket,
    'funke/fib25/training/tstvol-520-2.zarr')
