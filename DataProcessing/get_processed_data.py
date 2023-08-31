import os
import re

from azure.storage.blob import BlobServiceClient
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

connection_string = os.environ.get("CONNECTION_STRING")
blob_service_client = BlobServiceClient.from_connection_string(connection_string)

container_name = os.environ.get("CONTAINER_NAME")

# list blobs in the container
blobs = blob_service_client.get_container_client(container_name).list_blobs()

# filter for the processed files
processed_blobs = [blob.name for blob in blobs if "_processed.json" in blob.name]

# extract the datetime from the filenames and sort
sorted_blobs = sorted(
    processed_blobs,
    key=lambda x: re.search(r"(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})", x).group(),
    reverse=True    # latest first
)

# get the latest processed blob
blob_name = sorted_blobs[0] if sorted_blobs else None

# Download the blob to a local file
local_filename = blob_name.split("/")[-1]

data_folder_path = os.path.join(os.getcwd(), 'data')
filepath = os.path.join(data_folder_path, local_filename)

blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)

with open(filepath, "wb") as download_file:
    download_file.write(blob_client.download_blob().readall())

print(f"Downloaded {blob_name} to {filepath}")
