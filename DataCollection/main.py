import utils
import json
import datetime
import os

from googleapiclient.discovery import build
from config import API_KEY
from blob_upload import upload_to_blob

youtube = build('youtube', 'v3', developerKey=API_KEY)

CHANNEL_IDS = [
    utils.get_channel_id('CNN'),  # CNN
    utils.get_channel_id('BBCNews'),  # BBC
    utils.get_channel_id('aljazeeraenglish'),  # Al Jazeera English
    utils.get_channel_id('FoxNews'),  # Fox News
    utils.get_channel_id('guardiannews'),  # The Guardian
    utils.get_channel_id('indiatoday'),  # India Today
    'UCqnbDFdCpuN8CMEg0VuEBqA',  # New York Times
    'UC4SUWizzKc1tptprBkWjX2Q',  # South China Morning Post
    utils.get_channel_id('Channel4News'),  # Channel 4 News
    utils.get_channel_id('abcnewsaustralia')  # ABC News Australia
]

all_videos = []
# for each channel id, get 10 of the latest videos by id and title
for channel_id in CHANNEL_IDS:
    all_videos.extend(utils.get_videos_from_channel(channel_id))

all_comments = []
# for each video extracted, get 100 comments
for video in all_videos:
    all_comments.extend(utils.get_comments(video['video_id']))


def save_comments(comments, filename):
    with open(filename, 'w') as f:
        json.dump(comments, f)


data_folder_path = os.path.join(os.getcwd(), 'DataCollection/data')
timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
filename = f'comments_data_{timestamp}.json'
filepath = os.path.join(data_folder_path, filename)
save_comments(all_comments, filepath)


# Use the function
account_url = "https://jacquelinew945.blob.core.windows.net/youtube-news-comments"
container_name = "youtube-news-comments"
blob_name = os.path.basename(filepath)
sas_token = "YOUR_SAS_TOKEN"

upload_to_blob(account_url, container_name, blob_name, sas_token, filepath)
