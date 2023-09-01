import json
import datetime
import os
import utils

from dotenv import load_dotenv, find_dotenv
from googleapiclient.discovery import build
from blob_upload import upload_to_blob

load_dotenv(find_dotenv())

# get environment variables
api_key = os.environ.get("API_KEY")
account_url = os.environ.get("ACCOUNT_URL")
container_name = os.environ.get("CONTAINER_NAME")
sas_token = os.environ.get("SAS_TOKEN")

youtube = build('youtube', 'v3', developerKey=api_key)

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
# for each video extracted, get 100 comments and append the video title
for video in all_videos:
    comments_for_video = utils.get_comments(video['video_id'])
    for comment in comments_for_video:
        comment['video_title'] = video['title']  # append the video title to the comment data
    all_comments.extend(comments_for_video)


def save_comments(comments, filename):
    with open(filename, 'w') as f:
        json.dump(comments, f)


data_folder_path = os.path.join(os.getcwd(), 'data')
timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
filename = f'comments_data_{timestamp}.json'
filepath = os.path.join(data_folder_path, filename)
save_comments(all_comments, filepath)

blob_name = filename

upload_to_blob(account_url, container_name, blob_name, sas_token, filepath)
