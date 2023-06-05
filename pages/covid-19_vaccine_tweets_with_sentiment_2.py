import streamlit as st

# OVERRIDE
print = st.write

# Import Libraries
from textblob import TextBlob
import sys
import tweepy
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import nltk
import pycountry
import re
import string
import csv

from wordcloud import WordCloud, STOPWORDS
from PIL import Image
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from langdetect import detect
from nltk.stem import SnowballStemmer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import CountVectorizer
nltk.download('vader_lexicon')

file = open('./static/covid-19_vaccine_tweets_with_sentiment2.csv')
keyword ='covid_19'
csv.reader(file)
df = pd.read_csv(file, sep=';')
# adding a row_id field to the dataframe
df["row_id"] = df.index + 1
#print first 10 rows
print (df.head(10))

#create a new data frame with "id" and "tweet_text" fields
df_subset = df[['row_id', 'tweet_text']].copy()
#data clean-up
#remove all non-aphabet characters
df_subset['tweet_text'] = df_subset['tweet_text'].str.replace("[^a-zA-Z#]", " ")
#covert to lower-case
df_subset['tweet_text'] = df_subset['tweet_text'].str.casefold()
print (df_subset.head(10))

# set up empty dataframe for staging output
df1=pd.DataFrame()
df1['row_id']=['99999999999']
df1['sentiment_type']='NA999NA'
df1['sentiment_score']=0


def percentage(part, whole):
    return 100 * float(part) / float(whole)


print('Processing sentiment analysis...')
noOfTweet = 0
sid = SentimentIntensityAnalyzer()
t_df = df1
for index, row in df_subset.iterrows():
    scores = sid.polarity_scores(row[1])
    noOfTweet += 1
    for key, value in scores.items():
        temp = [key, value, row[0]]
        df1['row_id'] = row[0]
        df1['sentiment_type'] = key
        df1['sentiment_score'] = value
        # t_df=t_df.append(df1)
        t_df = pd.concat([t_df, df1])
# remove dummy row with row_id = 99999999999
t_df_cleaned = t_df[t_df.row_id != '99999999999']

# remove duplicates if any exist
t_df_cleaned = t_df_cleaned.drop_duplicates()
# only keep rows where sentiment_type = compound
t_df_cleaned = t_df[t_df.sentiment_type == 'compound']
print(t_df_cleaned.head(10))

positive = 0
negative = 0
neutral = 0
polarity = 0
tweet_list = []
neutral_list = []
negative_list = []
positive_list = []

for index, row in df_subset.iterrows():
    tweet_list.append(row[1])
    analysis = TextBlob(row[1])
    score = SentimentIntensityAnalyzer().polarity_scores(row[1])
    neg = score['neg']
    neu = score['neu']
    pos = score['pos']
    comp = score['compound']
    polarity += analysis.sentiment.polarity

    if neg > pos:
        negative_list.append(row[1])
        negative += 1
    elif pos > neg:
        positive_list.append(row[1])
        positive += 1

    elif pos == neg:
        neutral_list.append(row[1])
        neutral += 1

positive = percentage(positive, noOfTweet)
negative = percentage(negative, noOfTweet)
neutral = percentage(neutral, noOfTweet)
polarity = percentage(polarity, noOfTweet)
positive = format(positive, '.1f')
negative = format(negative, '.1f')
neutral = format(neutral, '.1f')

#Number of Tweets (Total, Positive, Negative, Neutral)
tweet_list = pd.DataFrame(tweet_list)
neutral_list = pd.DataFrame(neutral_list)
negative_list = pd.DataFrame(negative_list)
positive_list = pd.DataFrame(positive_list)
print("Total Tweets: ",len(tweet_list))
print("Positive Tweets: ",len(positive_list))
print("negative Tweets: ", len(negative_list))
print("neutral Tweets: ",len(neutral_list))

#Creating PieCart
keyword ='covid_19'
labels = ['Positive ['+str(positive)+'%]' , 'Neutral ['+str(neutral)+'%]','Negative ['+str(negative)+'%]']
sizes = [positive, neutral, negative]
colors = ['yellowgreen', 'blue','red']
fig, ax = plt.subplots()
patches, texts = ax.pie(sizes,colors=colors, startangle=90)
plt.style.use('default')
ax.legend(labels)
ax.set_title("Sentiment Analysis Result for keyword= "+keyword+" " )
ax.axis('equal')
st.pyplot(fig)
# patches, texts = plt.pie(sizes,colors=colors, startangle=90)
# plt.style.use('default')
# plt.legend(labels)
# plt.title("Sentiment Analysis Result for keyword= "+keyword+" " )
# plt.axis('equal')
# plt.show()

# create data for Pie Chart
# pichart = count_values_in_column(tw_list,"sentiment")

size = [positive, neutral, negative]

# Create a circle for the center of the plot
import matplotlib.patches as mpatches
fig, ax = plt.subplots()
my_circle = mpatches.Circle((0, 0), 0.7, color='white')
ax.pie(size, labels=['Positive [' + str(positive) + '%]', 'Neutral [' + str(neutral) + '%]',
                      'Negative [' + str(negative) + '%]'], colors=['green', 'blue', 'red'])

ax.add_artist(my_circle)
st.pyplot(fig)
# p = plt.gcf()
# p.gca().add_artist(my_circle)
# plt.show()

#merge dataframes
df_output = pd.merge(df, t_df_cleaned, on='row_id', how='inner')
print(df_output.head(10))

df_output[["sentiment_score"]].describe()

#generate mean of sentiment_score by label
dfg = df_output.groupby(['label'])['sentiment_score'].mean()
#create a bar plot
dfg.plot(kind='bar', title='Sentiment Score', ylabel='Mean Sentiment Score',
         xlabel='label', figsize=(6, 5))

#Creating new dataframe and new features
tw_list = pd.DataFrame(tweet_list)
tw_list["text"] = tw_list[0]

#Removing RT, Punctuation etc
remove_rt = lambda x: re.sub('RT @\w+: '," ",x)
rt = lambda x: re.sub("[^a-zA-Z#]", " ",x)
tw_list["text"] = tw_list.text.map(remove_rt).map(rt)
tw_list["text"] = tw_list.text.str.lower()
tw_list.head(10)

# Calculating Negative, Positive, Neutral and Compound values
tw_list[['polarity', 'subjectivity']] = tw_list['text'].apply(lambda Text: pd.Series(TextBlob(Text).sentiment))
tw_list.head(10)
for index, row in tw_list['text'].items():
    score = SentimentIntensityAnalyzer().polarity_scores(row)
    neg = score['neg']
    neu = score['neu']
    pos = score['pos']
    comp = score['compound']
    if neg > pos:
        tw_list.loc[index, 'sentiment'] = "negative"
    elif pos > neg:
        tw_list.loc[index, 'sentiment'] = "positive"
    else:
        tw_list.loc[index, 'sentiment'] = "neutral"

    tw_list.loc[index, 'neg'] = neg
    tw_list.loc[index, 'neu'] = neu
    tw_list.loc[index, 'pos'] = pos
    tw_list.loc[index, 'compound'] = comp
tw_list.head(10)

#Creating new data frames for all sentiments (positive, negative and neutral)
tw_list_negative = tw_list[tw_list["sentiment"]=="negative"]
tw_list_positive = tw_list[tw_list["sentiment"]=="positive"]
tw_list_neutral = tw_list[tw_list["sentiment"]=="neutral"]

def count_values_in_column(data,feature):
 total=data.loc[:,feature].value_counts(dropna=False)
 percentage=round(data.loc[:,feature].value_counts(dropna=False,normalize=True)*100,2)
 return pd.concat([total,percentage],axis=1,keys=['Total','Percentage'])
#Count_values for sentiment
count_values_in_column(tw_list,"sentiment")



tweet_list
positive_list
negative_list
neutral_list

#Function to Create Wordcloud
def create_wordcloud(text):
    mask = np.array(Image.open("./static/cloud.png"))
    stopwords = set(STOPWORDS)
    wc = WordCloud(background_color="white",
    mask = mask,
    max_words=3000,
    stopwords=stopwords,
    repeat=True)
    wc.generate(str(text))
    wc.to_file("wc.png")
    print("Word Cloud Saved Successfully")
    path="wc.png"
    display=st.image
    display(Image.open(path))

# Creating wordcloud for all tweets
print('Processing wordcloud for all tweets...')
create_wordcloud(tw_list["text"].values)

# Creating wordcloud for positive sentiment
print('Processing wordcloud for positive tweets...')
create_wordcloud(tw_list_positive["text"].values)

# Creating wordcloud for negative sentiment
print('Processing wordcloud for negative tweets...')
create_wordcloud(tw_list_negative["text"].values)

# Calculating tweet’s lenght and word count
tw_list['text_len'] = tw_list['text'].astype(str).apply(len)
tw_list['text_word_count'] = tw_list['text'].apply(lambda x: len(str(x).split()))
round(pd.DataFrame(tw_list.groupby("sentiment").text_len.mean()), 2)

round(pd.DataFrame(tw_list.groupby("sentiment").text_word_count.mean()), 2)

nltk.download('stopwords')

 # Removing Punctuation
def remove_punct(text):
    text = "".join([char for char in text if char not in string.punctuation])
    text = re.sub('[0–9]+', '', text)
    return text

tw_list['punct'] = tw_list['text'].apply(lambda x: remove_punct(x))

# Appliyng tokenization
def tokenization(text):
    text = re.split('\W+', text)
    return text

tw_list['tokenized'] = tw_list['punct'].apply(lambda x: tokenization(x.lower()))
# Removing stopwords
stopword = nltk.corpus.stopwords.words('english')

def remove_stopwords(text):
    text = [word for word in text if word not in stopword]
    return text

tw_list['nonstop'] = tw_list['tokenized'].apply(lambda x: remove_stopwords(x))
# Appliyng Stemmer
ps = nltk.PorterStemmer()

def stemming(text):
    text = [ps.stem(word) for word in text]
    return text

tw_list['stemmed'] = tw_list['nonstop'].apply(lambda x: stemming(x))

# Cleaning Text
def clean_text(text):
    text_lc = "".join([word.lower() for word in text if word not in string.punctuation])  # remove puntuation
    text_rc = re.sub('[0-9]+', '', text_lc)
    tokens = re.split('\W+', text_rc)  # tokenization
    text = [ps.stem(word) for word in tokens if word not in stopword]  # remove stopwords and stemming
    return text

tw_list.head()

# Appliyng Countvectorizer
countVectorizer = CountVectorizer(analyzer=clean_text)
countVector = countVectorizer.fit_transform(tw_list['text'])
print('{} Number of reviews has {} words'.format(countVector.shape[0], countVector.shape[1]))

count_vect_df = pd.DataFrame(countVector.toarray(), columns=countVectorizer.get_feature_names_out())
count_vect_df.head()

# Most Used Words
count = pd.DataFrame(count_vect_df.sum())
countdf = count.sort_values(0, ascending=False).head(20)
countdf[1:11]

# Function to ngram
def get_top_n_gram(corpus, ngram_range, n=None):
    vec = CountVectorizer(ngram_range=ngram_range, stop_words='english').fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0)
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
    return words_freq[:n]

# n2_bigram
n2_bigrams = get_top_n_gram(tw_list['text'], (2, 2), 20)
st.table(n2_bigrams)

# n3_trigram
n3_trigrams = get_top_n_gram(tw_list['text'], (3, 3), 20)
st.table(n3_trigrams)

# n4_gram
n4_grams = get_top_n_gram(tw_list['text'], (4, 4), 20)
st.table(n4_grams)

def predicted(value):
    if value <= -0.5:
        return 1
    elif value <= 0.5:
        return 2
    else:
        return 3

df_output['predicted'] = df_output['sentiment_score'].map(predicted)
print(df_output.head(10))

# def decode_sentiment(score, include_neutral=True):
#      if include_neutral:
#          label = NEUTRAL
#          if score <= SENTIMENT_THRESHOLDS[0]:
#              label = NEGATIVE
#          elif score >= SENTIMENT_THRESHOLDS[1]:
#              label = POSITIVE
#
#          return label
#      else:
#          return NEGATIVE if score < 0.5 else POSITIVE
#
# y_pred_1d = []
# y_test_1d = list(df_output.label)
# scores = df_output.sentiment_score  # model.predict(x_test, verbose=1, batch_size=8000)
# y_pred_1d = [decode_sentiment(sentiment_score, include_neutral=False) for sentiment_score in scores]
#
# def plot_confusion_matrix(cm, classes,
#                            title='Confusion matrix',
#                            cmap=plt.cm.Blues):
#      """
#      This function prints and plots the confusion matrix.
#      Normalization can be applied by setting `normalize=True`.
#      """
#
#      cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
#
#      plt.imshow(cm, interpolation='nearest', cmap=cmap)
#      plt.title(title, fontsize=30)
#      plt.colorbar()
#      tick_marks = np.arange(len(classes))
#      plt.xticks(tick_marks, classes, rotation=90, fontsize=22)
#      plt.yticks(tick_marks, classes, fontsize=22)
#
#      fmt = '.2f'
#      thresh = cm.max() / 2.
#      for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
#          plt.text(j, i, format(cm[i, j], fmt),
#                   horizontalalignment="center",
#                   color="white" if cm[i, j] > thresh else "black")
#
#      plt.ylabel('True label', fontsize=25)
#      plt.xlabel('Predicted label', fontsize=25)
#
# cnf_matrix = confusion_matrix(y_test_1d, y_pred_1d)
# plt.figure(figsize=(12, 12))
# plot_confusion_matrix(cnf_matrix, classes=df_train.target.unique(), title="Confusion matrix")
# plt.show()
#
