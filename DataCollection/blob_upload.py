from azure.storage.blob import BlobServiceClient


"""
    Uploads a file to Azure Blob Storage
    
    Parameters:
        - account_url (str): URL of blob service account
        - container (str): Container name
        - blob (str): Name of blob file in container
        - sas_token (str): Shared Access Signature (SAS) authentication token
        - filepath (str): Local filepath of JSON file
   
"""


def upload_to_blob(account_url, container, blob, sas_token, filepath):
    # instantiate a BlobServiceClient using a SAS token
    blob_service_client = BlobServiceClient(account_url=account_url, credential=sas_token)

    # instantiate a ContainerClient to interact with a specific container
    container_client = blob_service_client.get_container_client(container)

    # instantiate a new BlobClient object to interact with a specific blob
    blob_client = container_client.get_blob_client(blob)

    # upload the file to the blob
    with open(filepath, 'rb') as f:
        blob_client.upload_blob(f)
