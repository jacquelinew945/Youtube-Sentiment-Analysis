import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt

from nltk.stem import WordNetLemmatizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from gensim.models import Word2Vec
from dotenv import load_dotenv, find_dotenv
from sklearn.cluster import KMeans
from collections import Counter
from wordcloud import WordCloud
from blob_upload import upload_to_blob

load_dotenv(find_dotenv())

# get environment variables
api_key = os.environ.get("API_KEY")
account_url = os.environ.get("ACCOUNT_URL")
container_name = os.environ.get("CONTAINER_NAME")
sas_token = os.environ.get("SAS_TOKEN")

file_pattern = 'comments_data_'

# list all files
all_files = [f for f in os.listdir('data') if f.startswith(file_pattern) and f.endswith('_processed.csv')]

# sort by timestamp
sorted_files = sorted(all_files, key=lambda x: x.split('_')[2:4], reverse=True)  # sorts by date and time

latest_file = None
if sorted_files:
    latest_file = os.path.join('data', sorted_files[0])

df = pd.read_csv(latest_file, delimiter='|')
# dataframe with only channel name, video id and title for merging purposes
df_titles = df[['channel_name', 'video_id', 'video_title']]

# track how many nans in a video before dropping rows with nan values
nan_rows = df[df['text'].isna()]
nan_counts = nan_rows.groupby('video_id').size()
df = df.dropna(subset=['text'])
# remove unwanted symbols from title column
df['video_title'] = df['video_title'].str.replace(r'[^\w\s]', '', regex=True)


def remove_consecutive_commas(s):
    s = str(s)
    s = s.strip(',')
    while ',,' in s:
        s = (s.replace(',,', ','))
    return s


df['filtered_words'] = df['filtered_words'].apply(remove_consecutive_commas)
df['filtered_words'] = df['filtered_words'].apply(lambda x: x.split(','))

# lemmatize the filtered words
lemmatizer = WordNetLemmatizer()


def lemmatize_tokens(tokens):
    return [lemmatizer.lemmatize(token) for token in tokens]


df['lemmatized_words'] = df['filtered_words'].apply(lemmatize_tokens)

# create a Word2Vector model
all_lemmatized_tokens = df['lemmatized_words'].tolist()
w2v_model = Word2Vec(sentences=all_lemmatized_tokens, vector_size=100, window=5, min_count=1, workers=4)


def text_to_vec(tokens, model):
    vecs = [model.wv[token] for token in tokens if token in model.wv.index_to_key]
    if len(vecs) > 0:
        return np.mean(vecs, axis=0)
    else:
        return np.zeros(model.vector_size)


# apply word2vec model to df
df['text_vector'] = df['lemmatized_words'].apply(lambda x: text_to_vec(x, w2v_model))

# prepare the data for Azure ML
agg_vec = df.groupby('video_id')['text_vector'].apply(lambda x: np.mean(np.array(x.tolist()), axis=0))
X = np.array(agg_vec.tolist())

# k-means clustering
n_clusters = 8
kmeans = KMeans(n_clusters=n_clusters, n_init=10)
clusters = kmeans.fit_predict(X)

# check cluster sizes
video_ids = agg_vec.index.values
df_clusters = pd.DataFrame({
    'video_id': video_ids,
    'cluster': clusters
})

# tidy up cluster dataframe
df_clusters = df_clusters.merge(df_titles, on='video_id', how='left')
df_clusters = df_clusters.drop_duplicates(subset=['video_id', 'video_title'])
df_clusters = df_clusters.sort_values(by='cluster')

print(df_clusters['cluster'].value_counts())
df = df.merge(df_clusters[['cluster', 'video_id']], on='video_id', how='left')


def top_words(cluster_id, n_words=20):
    comments = df[df['cluster'] == cluster_id]['lemmatized_words']
    all_words = [word for comment in comments for word in comment]
    most_common = Counter(all_words).most_common(n_words)
    return most_common


# add top words for each cluster to cluster dataframe
for cluster_id in df_clusters['cluster'].unique():
    print(f"Cluster {cluster_id} top words: {top_words(cluster_id)}")
    df_clusters['top_words'] = df_clusters['cluster'].map(top_words)

df = df.sort_values(by='cluster')


# by video_id
def top_words_and_title_and_channel_in_video(video_id, n_words=8):
    video_comments = df[df['video_id'] == video_id]['lemmatized_words']
    all_words = [word for comment_list in video_comments for word in comment_list]
    most_common = Counter(all_words).most_common(n_words)
    title = df[df['video_id'] == video_id]['video_title'].iloc[0]  # Getting the title of the video
    channel = df[df['video_id'] == video_id]['channel_name'].iloc[0]  # Getting the channel name of the video
    return title, most_common, channel


# add top words by video id
video_ids = df['video_id'].unique()
video_data = [top_words_and_title_and_channel_in_video(video_id) for video_id in video_ids]

# extract titles, top words, and channel names separately from the combined list
video_titles = [item[0] for item in video_data]
video_top_words = [item[1] for item in video_data]
video_channels = [item[2] for item in video_data]

# new dataframe
df_video = pd.DataFrame({
    'video_id': video_ids,
    'video_title': video_titles,
    'top_words': video_top_words,
    'channel_name': video_channels
})


# initialize sentiment analyzer
sia = SentimentIntensityAnalyzer()

# apply sentiment scores
df['lemmatized_words'] = df['lemmatized_words'].apply(lambda x: ' '.join(x))
df['sentiment_scores'] = df['lemmatized_words'].apply(lambda x: sia.polarity_scores(x)['compound'])
df['sentiment'] = df['sentiment_scores'].apply(lambda x: 'positive' if x > 0 else ('neutral' if x == 0 else 'negative'))

# percentage of comments which are positive in all videos
videos = []
for i in range(0, df.video_id.nunique()):
    positive = df[(df.video_id == df.video_id.unique()[i]) & (df.sentiment == 'positive')].count()[0]
    total = df[df.video_id == df.video_id.unique()[i]]['sentiment'].value_counts().sum()
    percentage = (positive/total) * 100
    videos.append(round(percentage, 2))

df_positivity = pd.DataFrame(videos, df.video_id.unique()).reset_index()
df_positivity.columns = ['video_id', 'positive_percentage']
df_positivity = df_positivity.merge(df_titles, on='video_id', how='inner')
df_positivity['comment_count'] = df_positivity.groupby('video_id')['video_id'].transform('size')
df_positivity = df_positivity.drop_duplicates()

# sort df by likes in descending order
# add the top 3 liked comments from each video as a new column
df_sorted = df.sort_values(by=['video_id', 'like_count'], ascending=[True, False])
top_3_comments = df_sorted.groupby('video_id').head(3).copy()
top_3_comments['top_liked'] = top_3_comments.apply(lambda row: (row['like_count'], row['text']), axis=1)
grouped_comments = top_3_comments.groupby('video_id')['top_liked'].apply(list).reset_index()
df_positivity = df_positivity.merge(grouped_comments, on='video_id', how='left')

# construct a wordcloud for all words
all_words = ' '.join([text for text in df['lemmatized_words']])
wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(all_words)

plt.figure(figsize=(10, 7))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
plt.show()

# construct a wordcloud for positive words
all_words_pos = ' '.join([text for text in df['lemmatized_words'][df.sentiment == 'positive']])

wordcloud_pos = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(all_words_pos)

plt.figure(figsize=(10, 7))
plt.imshow(wordcloud_pos, interpolation="bilinear")
plt.axis('off')
plt.show()

# construct a wordcloud for negative words
all_words_neg = ' '.join([text for text in df['lemmatized_words'][df.sentiment == 'negative']])

wordcloud_neg = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(all_words_neg)

plt.figure(figsize=(10, 7))
plt.imshow(wordcloud_neg, interpolation="bilinear")
plt.axis('off')
plt.show()
