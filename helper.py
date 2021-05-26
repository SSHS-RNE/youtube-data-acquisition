from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import requests
import urllib
import copy
import re
import json
from tqdm import tqdm
from typing import List, Dict, Any, Tuple, Final
import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import nltk
from datetime import datetime
from dateutil.parser import parse
from wordcloud import WordCloud

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))


API_KEY: Final[str] = 'AIzaSyB2OJynYIkD7nW7ymSGtmkSHp9iMVN1K-M'  # API 요청을 위한 키
BASE_URL: Final[str] = 'https://www.googleapis.com/youtube/v3/'  # url prefix


# api 엔드포인트에 요청을 보내는 헬퍼 함수
def retrieve_api(url: str, params={}, data={}, headers={}, method='GET'):
    params = copy.deepcopy(params)
    params.update({
        'key': API_KEY
    })

    return requests.request(method, urllib.parse.urljoin(BASE_URL, url), params=params, data=data, headers=headers)


# 비디오 ID를 받아, 비디오의 통계(조회수, 좋아요 수 등)을 반환한다.
def retrieve_statistics(videoId: str) -> List[Dict[str, int]]:
    resp = retrieve_api('videos', params={
        'id': videoId,
        'part': 'statistics'
    })
    assert(resp.ok)

    return json.loads(resp.text)['items'][0]['statistics']


# 비디오 ID를 받아, 모든 댓글을 반환한다.
def retrieve_comments(videoId: str) -> List[List[str]]:
    def extract_commentThread_text(item): return item['snippet']['topLevelComment']['snippet']['textDisplay']
    def extract_commentThread_timestamp(item): return item['snippet']['topLevelComment']['snippet']['publishedAt']
    
    pageToken = None
    result = []
    timestamp = []

    while True:
        resp = retrieve_api('commentThreads', params={
            'videoId': videoId,
            'part': 'snippet',
            'order': 'relevance',
            'pageToken': pageToken,
            'textFormat': 'plainText'
        })
        assert(resp.ok)

        commentData = json.loads(resp.text)

        if not "nextPageToken" in commentData:
            break
        pageToken = commentData["nextPageToken"]

        result += [*map(extract_commentThread_text, commentData['items'])]
        timestamp += [*map(extract_commentThread_timestamp,
                           commentData['items'])]

        print(len(result), end=' ')

    print('Done')
    return result, timestamp



# 비디오 ID를 받아 자막을 반환한다.
def retrieve_captions(videoId: str) -> List[List[str]]:
    def extract_caption_text(item): return re.sub('<(.|\n)*?>', '', item.text)
    
    resp = requests.get(
        f'https://video.google.com/timedtext?lang=en&v={videoId}')
    assert(resp.ok)

    captionData = ET.fromstring(resp.text)
    captions = []
    for text in map(extract_caption_text, [*captionData]):
        if len(captions) == 0 or captions[-1] != text:
            captions.append(text)

    return captions


# 채널 ID를 받아 최근 비디오 50개의 의 정보를 반환한다. 반환형은 API 참조.
def get_recent_videos(channelId: str) -> List[Any]:
    resp = retrieve_api('search', params={
        'part': 'snippet',
        'channelId': channelId,
        'type': 'video',
        'maxResults': 50,
        'order': 'date'
    })

    return json.loads(resp.text)


# 플레이리스트 ID를 받아 앞 50개 비디오의 ID의 리스트를 반환한다.
def retrieve_playlist_videos(playlistId: str) -> List[str]:
    def extract_playlistItems_videoId(item): return item['snippet']['resourceId']['videoId']

    resp = retrieve_api('playlistItems', params={
        'part': 'snippet',
        'maxResults': 50,
        'playlistId': playlistId
    })

    return [*map(extract_playlistItems_videoId, json.loads(resp.text)['items'])]


# 제시된 문자열을 전처리하여 의미 있는 단어의 리스트를 반환한다.
def get_words(st: str) -> List[str]:
    st = re.sub('[^a-zA-Z\ ]', ' ', st)  # 공백, a-z, A-Z만 남딤
    result = word_tokenize(st.lower())  # 소문자로 바꾸고, 토큰화
    result = [*filter(lambda x: x not in stop_words, result)]  # stop words 제거
    result = [*map(lambda x: lemmatizer.lemmatize(x), result)]  # 표제어 추출

    return result


# 간단한 multiset 구현
def count_at_dict(dt: Dict[str, int], vl: str) -> None:
    if vl in dt:
        dt[vl] += 1
    else:
        dt[vl] = 1


# 문자열의 리스트를 받아 빈도수 데이터프레임을 반환한다.
def get_freq(strList: List[str]) -> pd.DataFrame:
    # 모든 단어에 빈도수 저장
    dt = dict()
    for tokenized in map(get_words, strList):
        for word in tokenized:
            count_at_dict(dt, word)

    # pandas.DataFrame으로 변환
    word = []
    freq = []
    for key in dt:
        word.append(key)
        freq.append(dt[key])

    df = pd.DataFrame.from_dict({
        'word': word,
        'freq': freq
    })

    # 빈도수 내림차순으로 정렬
    return df.sort_values(by='freq', ascending=False)


# 빈도수 데이터프레임을 받아 긍정적, 부정적 단어의 빈도의 튜플을 반환한다.
def get_posneg_freq(df: pd.DataFrame, positiveWords: List[str], negativeWords: List[str]) -> Tuple[int, int]:
    positiveCount = 0
    positiveIndex = 0
    positiveLength = len(positiveWords)

    negativeCount = 0
    negativeIndex = 0
    negativeLength = len(negativeWords)

    for _, rowSeries in df.sort_values(by='word').iterrows():
        if positiveIndex == positiveLength and negativeIndex == negativeLength:
            break

        while positiveIndex < positiveLength and positiveWords[positiveIndex] < rowSeries['word']:
            positiveIndex += 1
        if positiveIndex < positiveLength and positiveWords[positiveIndex] == rowSeries['word']:
            positiveCount += rowSeries['freq']

        while negativeIndex < negativeLength and negativeWords[negativeIndex] < rowSeries['word']:
            negativeIndex += 1
        if negativeIndex < negativeLength and negativeWords[negativeIndex] == rowSeries['word']:
            negativeCount += rowSeries['freq']

    return (positiveCount, negativeCount)


# 빈도수 데이터프레임을 받아 Word Cloud를 만들고, 저장한다.
def draw_wordcloud(df: pd.DataFrame):
    wordcloud = WordCloud(background_color='white', width=960, height=540, max_font_size=150).generate_from_frequencies(
        {row[1]['word']: row[1]['freq'] for row in df[:100].iterrows()})

    plt.figure(figsize=(16, 9))
    plt.axis('off')
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.savefig('wordcloud.svg')


# word-time 데이터프레임을 받아 시간별 빈도수 벡터를 반환한다.
def get_freq_vec(df: pd.DataFrame, w: str, bins: List[float]) -> List[float]:
    se = df[df['word'] == w]['time']
    result = np.zeros(len(bins)-1)
    for i in range(len(bins)-1):
        result[i] = len(se[(bins[i] <= se) & (se < bins[i+1])])

    return result / np.linalg.norm(result)