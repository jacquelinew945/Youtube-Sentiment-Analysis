from googleapiclient.discovery import build
from config import API_KEY

youtube = build('youtube', 'v3', developerKey=API_KEY)


# sample usage: get_channel_id('CNN')
def get_channel_id(username):
    response = youtube.channels().list(
        part='id',
        forUsername=username
    ).execute()

    # Extract the channel ID from the response if it exists
    channel_id = None
    if response.get('items'):
        channel_id = response['items'][0]['id']

    return channel_id


def get_videos_from_channel(channel_id, max_results=10):
    request = youtube.search().list(
        part="snippet",
        channelId=channel_id,
        maxResults=max_results,
        order="date",  # order by latest date
        type="video"
    )
    response = request.execute()
    return [{'video_id': item['id']['videoId'], 'title': item['snippet']['title']} for item in response['items']]


def get_comments(video_id, max_results=100):
    comments = []
    request = youtube.commentThreads().list(
        part="snippet",
        videoId=video_id,
        maxResults=max_results,
        order="relevance",
        textFormat="plainText"
    )
    response = request.execute()

    for item in response['items']:
        comment = item['snippet']['topLevelComment']['snippet']
        comments.append({
            'video_id': video_id,
            'comment_id': comment['id'],
            'text': comment['textDisplay'],
            'like_count': comment['likeCount'],
            'published_at': comment['publishedAt'],
            'author': comment['authorDisplayName'],
        })
    return comments
