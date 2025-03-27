import os
import io
import json
import datetime
import requests
import boto3
from typing import List, Dict
from google.oauth2.credentials import Credentials as GoogleCredentials
from googleapiclient.discovery import build
from msal import ConfidentialClientApplication
from abc import ABC, abstractmethod
from azure.storage.blob import BlobServiceClient, ContentSettings

###############################################################################
# Base Datalake Interface
###############################################################################
class DataLake(ABC):
    @abstractmethod
    def save_file_with_metadata(self, file_content: bytes, file_path: str, metadata: dict):
        """
        Stores the file bytes under 'file_path', plus writes a .metadata.json
        or equivalent store containing the 'metadata' dictionary.
        """
        pass

    @abstractmethod
    def load_file(self, file_path: str) -> bytes:
        """
        Retrieves raw bytes for the file located at 'file_path'.
        """
        pass


###############################################################################
# Local Datalake
###############################################################################
class LocalFileDataLake(DataLake):
    def __init__(self):
        self.base_path = os.environ.get("LOCAL_DATALAKE_PATH", "local_datalake")

    def save_file_with_metadata(self, file_content: bytes, file_path: str, metadata: dict):
        full_file_path = os.path.join(self.base_path, file_path)
        os.makedirs(os.path.dirname(full_file_path), exist_ok=True)

        # Save the file
        with open(full_file_path, 'wb') as f:
            f.write(file_content)

        # Save metadata to a separate JSON file
        meta_path = full_file_path + ".metadata.json"
        with open(meta_path, 'w', encoding='utf-8') as mf:
            json.dump(metadata, mf, ensure_ascii=False, indent=2)

    def load_file(self, file_path: str) -> bytes:
        full_file_path = os.path.join(self.base_path, file_path)
        with open(full_file_path, 'rb') as f:
            return f.read()


###############################################################################
# S3 Datalake (AWS)
###############################################################################
class S3DataLake(DataLake):
    def __init__(self):
        # Uses standard AWS env vars: AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, etc.
        self.s3 = boto3.client(
            's3',
            aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
            region_name=os.environ.get("AWS_DEFAULT_REGION")
        )
        # S3 bucket name for storing files
        self.bucket_name = os.environ.get("DATALAKE_BUCKET", "my-datalake-bucket")

    def save_file_with_metadata(self, file_content: bytes, file_path: str, metadata: dict):
        # Put the actual file
        self.s3.put_object(
            Bucket=self.bucket_name,
            Key=file_path,
            Body=file_content,
            Metadata={k: str(v) for k, v in metadata.items() if isinstance(v, (str, int, float))}
        )

        # Put the metadata as .metadata.json
        meta_path = file_path + ".metadata.json"
        self.s3.put_object(
            Bucket=self.bucket_name,
            Key=meta_path,
            Body=json.dumps(metadata).encode('utf-8')
        )

    def load_file(self, file_path: str) -> bytes:
        resp = self.s3.get_object(Bucket=self.bucket_name, Key=file_path)
        return resp['Body'].read()


###############################################################################
# MinIO Datalake (S3-Compatible)
###############################################################################
class MinioDataLake(DataLake):
    """
    Identical approach to S3, but uses the MINIO_ENDPOINT & MINIO_BUCKET environment variables.
    """
    def __init__(self):
        self.minio_client = boto3.client(
            's3',
            endpoint_url=os.environ.get("MINIO_ENDPOINT", "http://localhost:9000"),
            aws_access_key_id=os.environ.get("MINIO_ACCESS_KEY"),
            aws_secret_access_key=os.environ.get("MINIO_SECRET_KEY"),
            region_name="us-east-1",  # Typically not relevant for MinIO
            verify=False  # Disable SSL verification if self-signed
        )
        self.bucket_name = os.environ.get("MINIO_BUCKET", "my-minio-bucket")

    def save_file_with_metadata(self, file_content: bytes, file_path: str, metadata: dict):
        # Upload file
        self.minio_client.put_object(
            Bucket=self.bucket_name,
            Key=file_path,
            Body=file_content,
            Metadata={k: str(v) for k, v in metadata.items() if isinstance(v, (str, int, float))}
        )

        # Upload metadata
        meta_path = file_path + ".metadata.json"
        self.minio_client.put_object(
            Bucket=self.bucket_name,
            Key=meta_path,
            Body=json.dumps(metadata).encode('utf-8')
        )

    def load_file(self, file_path: str) -> bytes:
        resp = self.minio_client.get_object(Bucket=self.bucket_name, Key=file_path)
        return resp['Body'].read()


###############################################################################
# Azure Blob Datalake
###############################################################################
class AzureBlobDataLake(DataLake):
    def __init__(self):
        self.connection_string = os.environ.get("AZURE_BLOB_CONNECTION_STRING")
        self.container_name = os.environ.get("AZURE_BLOB_CONTAINER", "my-datalake-container")
        self.service_client = BlobServiceClient.from_connection_string(self.connection_string)
        self.container_client = self.service_client.get_container_client(self.container_name)

        # Ensure the container exists
        if not self.container_client.exists():
            self.container_client.create_container()

    def save_file_with_metadata(self, file_content: bytes, file_path: str, metadata: dict):
        # Upload main file
        blob_client = self.container_client.get_blob_client(blob=file_path)
        blob_client.upload_blob(
            file_content,
            overwrite=True,
            metadata={k: str(v) for k, v in metadata.items() if isinstance(v, (str, int, float))},
            content_settings=ContentSettings(content_type="application/octet-stream")
        )

        # Upload metadata as separate .metadata.json
        meta_path = file_path + ".metadata.json"
        meta_blob_client = self.container_client.get_blob_client(blob=meta_path)
        meta_blob_client.upload_blob(
            json.dumps(metadata).encode('utf-8'),
            overwrite=True,
            content_settings=ContentSettings(content_type="application/json")
        )

    def load_file(self, file_path: str) -> bytes:
        blob_client = self.container_client.get_blob_client(blob=file_path)
        downloader = blob_client.download_blob()
        return downloader.readall()


###############################################################################
# Data Lake Factory
###############################################################################
def get_datalake(datalake_type: str) -> DataLake:
    """
    Returns the appropriate datalake implementation based on datalake_type.
    Options: 's3', 'azureblob', 'local', 'minio'.
    """
    if datalake_type == 's3':
        return S3DataLake()
    elif datalake_type == 'azureblob':
        return AzureBlobDataLake()
    elif datalake_type == 'local':
        return LocalFileDataLake()
    elif datalake_type == 'minio':
        return MinioDataLake()
    else:
        raise ValueError(f"Unsupported datalake_type: {datalake_type}")


###############################################################################
# Google Drive Integration
###############################################################################
class GoogleDriveIntegration:
    def __init__(self):
        self.client_id = os.environ.get("GOOGLE_CLIENT_ID")
        self.client_secret = os.environ.get("GOOGLE_CLIENT_SECRET")
        self.refresh_token = os.environ.get("GOOGLE_REFRESH_TOKEN")
        self.token_uri = "https://oauth2.googleapis.com/token"
        self.scopes = ["https://www.googleapis.com/auth/drive.readonly"]
        self.service = self._authenticate()

    def _authenticate(self):
        from google.auth.transport.requests import Request
        creds = GoogleCredentials(
            token=None,
            refresh_token=self.refresh_token,
            token_uri=self.token_uri,
            client_id=self.client_id,
            client_secret=self.client_secret,
            scopes=self.scopes
        )
        creds.refresh(Request())
        return build("drive", "v3", credentials=creds)

    def list_files(self, folder_id: str = None) -> List[Dict]:
        query = f"'{folder_id}' in parents" if folder_id else None
        results = self.service.files().list(
            q=query,
            fields="files(id, name, mimeType, createdTime, owners)"
        ).execute()
        return results.get('files', [])

    def download_file(self, file_id: str) -> bytes:
        from googleapiclient.http import MediaIoBaseDownload
        request = self.service.files().get_media(fileId=file_id)
        fh = io.BytesIO()
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while not done:
            status, done = downloader.next_chunk()
        fh.seek(0)
        return fh.read()


###############################################################################
# OneDrive Integration
###############################################################################
class OneDriveIntegration:
    def __init__(self):
        self.client_id = os.environ.get("MS_CLIENT_ID")
        self.client_secret = os.environ.get("MS_CLIENT_SECRET")
        self.tenant_id = os.environ.get("MS_TENANT_ID")
        self.authority = f"https://login.microsoftonline.com/{self.tenant_id}"
        self.scope = ["https://graph.microsoft.com/.default"]
        self.app = ConfidentialClientApplication(
            client_id=self.client_id,
            client_credential=self.client_secret,
            authority=self.authority
        )
        self.token = self._get_token()

    def _get_token(self):
        result = self.app.acquire_token_silent(self.scope, account=None)
        if not result:
            result = self.app.acquire_token_for_client(scopes=self.scope)
        if "access_token" not in result:
            raise Exception("Could not obtain access token for OneDrive/SharePoint.")
        return result["access_token"]

    def list_files(self, drive_id: str, folder_path: str = None) -> List[Dict]:
        headers = {"Authorization": f"Bearer {self.token}"}
        if folder_path:
            url = f"https://graph.microsoft.com/v1.0/drives/{drive_id}/root:/{folder_path}:/children"
        else:
            url = f"https://graph.microsoft.com/v1.0/drives/{drive_id}/root/children"
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        return response.json().get('value', [])

    def download_file(self, drive_id: str, item_id: str) -> bytes:
        headers = {"Authorization": f"Bearer {self.token}"}
        url = f"https://graph.microsoft.com/v1.0/drives/{drive_id}/items/{item_id}"
        metadata_resp = requests.get(url, headers=headers)
        metadata_resp.raise_for_status()
        download_url = metadata_resp.json().get('@microsoft.graph.downloadUrl')
        if not download_url:
            raise Exception("No download URL found for the specified file.")
        file_resp = requests.get(download_url)
        file_resp.raise_for_status()
        return file_resp.content


###############################################################################
# SharePoint Integration
###############################################################################
class SharePointIntegration:
    def __init__(self):
        self.client_id = os.environ.get("MS_CLIENT_ID")
        self.client_secret = os.environ.get("MS_CLIENT_SECRET")
        self.tenant_id = os.environ.get("MS_TENANT_ID")
        self.authority = f"https://login.microsoftonline.com/{self.tenant_id}"
        self.scope = ["https://graph.microsoft.com/.default"]
        self.app = ConfidentialClientApplication(
            client_id=self.client_id,
            client_credential=self.client_secret,
            authority=self.authority
        )
        self.token = self._get_token()

    def _get_token(self):
        result = self.app.acquire_token_silent(self.scope, account=None)
        if not result:
            result = self.app.acquire_token_for_client(scopes=self.scope)
        if "access_token" not in result:
            raise Exception("Could not obtain access token for SharePoint.")
        return result["access_token"]

    def list_files(self, site_id: str, drive_id: str, folder_path: str = None) -> List[Dict]:
        headers = {"Authorization": f"Bearer {self.token}"}
        if folder_path:
            url = f"https://graph.microsoft.com/v1.0/sites/{site_id}/drives/{drive_id}/root:/{folder_path}:/children"
        else:
            url = f"https://graph.microsoft.com/v1.0/sites/{site_id}/drives/{drive_id}/root/children"
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        return response.json().get('value', [])

    def download_file(self, site_id: str, drive_id: str, item_id: str) -> bytes:
        headers = {"Authorization": f"Bearer {self.token}"}
        url = f"https://graph.microsoft.com/v1.0/sites/{site_id}/drives/{drive_id}/items/{item_id}"
        metadata_resp = requests.get(url, headers=headers)
        metadata_resp.raise_for_status()
        download_url = metadata_resp.json().get('@microsoft.graph.downloadUrl')
        if not download_url:
            raise Exception("No download URL found for the specified file.")
        file_resp = requests.get(download_url)
        file_resp.raise_for_status()
        return file_resp.content


###############################################################################
# S3Integration for Migrating from Another S3 Bucket
###############################################################################
class S3Integration:
    def __init__(self):
        # Another S3 client used for migrating from an external S3 bucket
        self.s3 = boto3.client(
            's3',
            aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
            region_name=os.environ.get("AWS_DEFAULT_REGION")
        )

    def list_files(self, bucket_name: str, prefix: str = "") -> List[Dict]:
        response = self.s3.list_objects_v2(Bucket=bucket_name, Prefix=prefix)
        if 'Contents' not in response:
            return []
        return [{"Key": obj["Key"]} for obj in response["Contents"]]

    def download_file(self, bucket_name: str, object_key: str) -> bytes:
        file_obj = io.BytesIO()
        self.s3.download_fileobj(bucket_name, object_key, file_obj)
        file_obj.seek(0)
        return file_obj.read()


###############################################################################
# Data Integration Method
###############################################################################
def integrate_data_into_datalake(provider: str, datalake_type: str, **kwargs):
    """
    Download files from an external provider (Google Drive, OneDrive, SharePoint, or another S3),
    then store them in the chosen local/remote data lake (e.g., minio, s3, azureblob, local).
    """
    datalake = get_datalake(datalake_type)
    now_str = datetime.datetime.utcnow().isoformat()

    if provider == "google_drive":
        drive = GoogleDriveIntegration()
        files = drive.list_files(folder_id=kwargs.get("folder_id"))
        for f in files:
            file_id = f['id']
            content = drive.download_file(file_id)
            metadata = {
                "provider": "google_drive",
                "file_id": file_id,
                "name": f.get("name"),
                "mimeType": f.get("mimeType"),
                "createdTime": f.get("createdTime"),
                "authors": [o.get('displayName') for o in f.get('owners', [])],
                "imported_at": now_str
            }
            path_name = f.get('name', file_id)
            path = f"google_drive/{path_name}"
            datalake.save_file_with_metadata(content, path, metadata)

    elif provider == "onedrive":
        od = OneDriveIntegration()
        drive_id = kwargs.get("drive_id")
        folder_path = kwargs.get("folder_path")
        files = od.list_files(drive_id, folder_path)
        for f in files:
            if 'file' in f:
                item_id = f['id']
                content = od.download_file(drive_id, item_id)
                metadata = {
                    "provider": "onedrive",
                    "drive_id": drive_id,
                    "item_id": item_id,
                    "name": f.get("name"),
                    "createdDateTime": f.get("createdDateTime"),
                    "lastModifiedDateTime": f.get("lastModifiedDateTime"),
                    "imported_at": now_str
                }
                path_name = f.get('name', item_id)
                path = f"onedrive/{path_name}"
                datalake.save_file_with_metadata(content, path, metadata)

    elif provider == "sharepoint":
        sp = SharePointIntegration()
        site_id = kwargs.get("site_id")
        drive_id = kwargs.get("drive_id")
        folder_path = kwargs.get("folder_path")
        files = sp.list_files(site_id, drive_id, folder_path)
        for f in files:
            if 'file' in f:
                item_id = f['id']
                content = sp.download_file(site_id, drive_id, item_id)
                metadata = {
                    "provider": "sharepoint",
                    "site_id": site_id,
                    "drive_id": drive_id,
                    "item_id": item_id,
                    "name": f.get("name"),
                    "createdDateTime": f.get("createdDateTime"),
                    "lastModifiedDateTime": f.get("lastModifiedDateTime"),
                    "imported_at": now_str
                }
                path_name = f.get('name', item_id)
                path = f"sharepoint/{path_name}"
                datalake.save_file_with_metadata(content, path, metadata)

    elif provider == "s3":
        # Migrate from an external AWS S3 bucket to your chosen data lake
        s3i = S3Integration()
        bucket_name = kwargs.get("bucket_name")
        prefix = kwargs.get("prefix", "")
        objects = s3i.list_files(bucket_name, prefix)
        for obj in objects:
            key = obj['Key']
            content = s3i.download_file(bucket_name, key)
            metadata = {
                "provider": "aws_s3",
                "source_bucket": bucket_name,
                "object_key": key,
                "imported_at": now_str
            }
            # Replace / with _ to avoid subfolder issues
            safe_key = key.replace('/', '_')
            path = f"aws_s3/{safe_key}"
            datalake.save_file_with_metadata(content, path, metadata)

    else:
        raise ValueError("Unsupported provider")
