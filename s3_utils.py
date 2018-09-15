from io import BytesIO
from pickle import dump, load
from pyarrow.feather import write_feather, read_feather
import boto3
from credentials import *


def write_to_s3(obj, bucket, file_name, type='pickle'):
    """
    Function to write python objects as pickle/feather to S3 in pickled format

    :param obj: python object to write
    :param bucket: S3 bucket name
    :param file_name: filname, including path
    :param type: Object type to write (pickle or feather)
    :return: None

    Example: write_to_s3(some_dict, 'kaggle-mercari', 'data/some_dict.pkl')
    """
    boto_client = boto3. \
        Session(aws_access_key_id=s3_keys['key'],
                aws_secret_access_key=s3_keys['secret_key']). \
        client('s3')

    with BytesIO() as f:
        if type == 'pickle':
            dump(obj, f)
        elif type == 'feather':
            write_feather(obj, f)
        else:
            raise ValueError('Invalid type argument. Expected pickle or feather')

        boto_client.put_object(Body=f.getvalue(), Bucket=bucket, Key=file_name)

    return


def read_from_s3(bucket, file_name, type='pickle'):
    """
    Function to read pickle/feather objects from S3

    :param bucket: S3 bucket name
    :param file_name: filname, including path
    :param type: Object type to read (pickle or feather)
    :return: Python object read

    Example: my_obj = read_from_s3(some_dict, 'kaggle-mercari', 'data/some_dict.pkl')
    """
    boto_client = boto3.\
        Session(aws_access_key_id=s3_keys['key'],
                aws_secret_access_key=s3_keys['secret_key']).\
        client('s3')

    obj = boto_client.get_object(Bucket=bucket, Key=file_name)

    with BytesIO(obj['Body'].read()) as f:
        if type == 'pickle':
            loaded_obj = load(f)
        elif type == 'feather':
            loaded_obj = read_feather(f)
        else:
            raise ValueError('Invalid type argument. Expected pickle or feather')

    return loaded_obj
