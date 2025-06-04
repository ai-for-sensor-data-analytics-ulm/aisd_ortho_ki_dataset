import os
from pathlib import Path
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

    def __bucket_exists(self, name: str) -> bool:
        """Check whether a bucket exists.

        Parameters
        ----------
        name : str
            Bucket name.

        Returns
        -------
        bool
            ``True`` if the bucket exists.
        """
        found = self.__minioClient.bucket_exists(name)
        return found

    def download_folder(self, folder_name: str, local_directory: str) -> None:
        """Download a folder and its files from MinIO.

        Parameters
        ----------
        folder_name : str
            Name of the folder in MinIO.
        local_directory : str
            Local directory to download the content to.
        """
        for obj in self.__minioClient.list_objects(self.bucket_name, prefix=folder_name, recursive=True):
            local_file_path = os.path.join(local_directory, obj.object_name)
            if not os.path.exists(os.path.dirname(local_file_path)):
                os.makedirs(os.path.dirname(local_file_path))
            with open(local_file_path, 'wb') as file_data:
                for d in self.__minioClient.get_object(self.bucket_name, obj.object_name):
                    file_data.write(d)

    def save_pickle(
        self,
        pickle_data: bytes,
        object_name: str,
        bucket_name: str | None = None,
        content_type: str = "application/octet-stream",
    ) -> None:
        """Save binary pickle data to MinIO.

        Parameters
        ----------
        pickle_data : bytes
            Data blob to upload.
        object_name : str
            Target object name.
        bucket_name : str, optional
            Destination bucket, defaults to the connector bucket.
        content_type : str, optional
            MIME type for the object.
        """
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

    def load_ml_dataset(
        self,
        minio_path: str,
        file_name: str,
        save_path: Path,
        bucket_name: str | None = None,
    ) -> tuple:
        """Load or download a machine learning dataset pickle.

        Parameters
        ----------
        minio_path : str
            Path in the bucket.
        file_name : str
            File name of the dataset.
        save_path : Path
            Local directory for caching.
        bucket_name : str, optional
            Bucket to use.

        Returns
        -------
        tuple
            ``(X, config)`` tuple loaded from the pickle file.
        """
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

    def download_hdf5_dataset(
        self,
        minio_path: str,
        file_name: str,
        save_path: Path,
        bucket_name: str | None = None,
    ) -> None:
        """Download an HDF5 dataset file if not present locally."""
        if not os.path.isfile(save_path / file_name):
            print('Attempting to download ', str(minio_path/file_name), ' to ', str(save_path/file_name))
            if bucket_name == None:
                bucket_name = self.bucket_name
            response = self.__minioClient.fget_object(bucket_name=bucket_name,
                                                      object_name=str(minio_path / file_name),
                                                      file_path=str(save_path / file_name),
                                                      progress=progress.Progress())
        return

    def save_hdf5_to_minio(
        self,
        hdf5_filepath: str,
        object_name: str,
        bucket_name: str,
    ) -> None:
        """Upload a local HDF5 file to MinIO."""
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


