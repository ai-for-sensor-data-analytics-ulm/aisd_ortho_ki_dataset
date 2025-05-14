import os
from minio import Minio
import io

class MinIOConnector:
    def __init__(self, host=None, access_key=None, secret_key=None, bucket_name=None, secure=False):
        if host is None:
            self.host = os.environ['AWS_S3_ENDPOINT_URL']
        else:
            self.host = host
        if access_key is None:
            self.access_key = os.environ['AWS_ACCESS_KEY_ID']
        else:
            self.access_key = access_key
        if secret_key is None:
            self.secret_key = os.environ['AWS_SECRET_ACCESS_KEY']
        else:
            self.secret_key = secret_key

        self.__minioClient = Minio(
            self.host,
            access_key=self.access_key,
            secret_key=self.secret_key,
            secure=secure,
        )
        self.bucket_name = bucket_name
        if not self.__bucket_exists(bucket_name):
            print('select correct bucket name!')
            self.list_buckets()

    def __bucket_exists(self, name):
        """
        This method checks if a bucket exists

        :return: Boolean (True = bucket exists)
        """
        found = self.__minioClient.bucket_exists(name)
        return found#

    def download_folder(self, folder_name, local_directory):
        """
        This method downloads a folder and its files from Minio

        :param folder_name: The name of the folder in Minio
        :param local_directory: The local directory to download the folder to
        """
        for obj in self.__minioClient.list_objects(self.bucket_name, prefix=folder_name, recursive=True):
            local_file_path = os.path.join(local_directory, obj.object_name)
            if not os.path.exists(os.path.dirname(local_file_path)):
                os.makedirs(os.path.dirname(local_file_path))
            with open(local_file_path, 'wb') as file_data:
                for d in self.__minioClient.get_object(self.bucket_name, obj.object_name):
                    file_data.write(d)

    def save_pickle(self, pickle_data, object_name, bucket_name=None, content_type="application/octet-stream"):
        if bucket_name == None:
            bucket_name = self.bucket_name

        data_stream = io.BytesIO(pickle_data)
        data_stream.seek(0)

        # put data as object into the bucket
        self.__minioClient.put_object(
            bucket_name=bucket_name,
            object_name=object_name,
            data=data_stream, length=len(pickle_data),
            content_type=content_type)
        return

    def load_ml_dataset(self, minio_path, file_name, save_path, bucket_name=None):
        if not os.path.isfile(save_path / file_name):
            print('Attempting to download ', str(minio_path/file_name), ' to ', str(save_path/file_name))
            if bucket_name == None:
                bucket_name = self.bucket_name
            response = self.__minioClient.get_object(bucket_name, str(minio_path / file_name))
            data = response.data
            data = pickle.loads(data)
            pickle.dump(data, open(save_path / file_name, 'wb'))
            del data
        print('loading data')
        dataset = pickle.load(open(save_path / file_name, 'rb'))
        return dataset['X'], dataset['config']

    def download_hdf5_dataset(self, minio_path, file_name, save_path, bucket_name=None):
        if not os.path.isfile(save_path / file_name):
            print('Attempting to download ', str(minio_path/file_name), ' to ', str(save_path/file_name))
            if bucket_name == None:
                bucket_name = self.bucket_name
            response = self.__minioClient.fget_object(bucket_name=bucket_name,
                                                      object_name=str(minio_path / file_name),
                                                      file_path=str(save_path / file_name),
                                                      progress=progress.Progress())
        return

    def save_hdf5_to_minio(self, hdf5_filepath, object_name, bucket_name):
        if bucket_name == None:
            bucket_name = self.bucket_name
        self.__minioClient.fput_object(bucket_name=bucket_name,
                                       object_name=object_name,
                                       file_path=hdf5_filepath,
                                       content_type='application/x-hdf5',
                                       progress=progress.Progress())
        return


minio_connector = MinIOConnector(host='imm-md-03:9000', access_key='minio', secret_key='mlgroup..11', bucket_name='ortho-ki')

data = minio_connector.download_folder('', '../raw_data/')


