import streamlit as st

"""
# Word Cloud - **word analytics**

## Objective
This interactive app illustrate an example of word analytics
and its related step － from cleaning to generating word cloud － to get
a meaningful word with the most occurrences.

## Dataset
[The dataset][the-dataset] used is taken from [kaggle][kaggle-dataset]. Created by [Gabriel Preda][gabriel-preda].
The dataset is a collection of tweets related to Covid-19 vaccines with
manually annotated sentiments (negative, neutral, positive).
Negative sentiment is labeled as 1, neutral as 2, and positive as 3.

[the-dataset]: app/static/covid-19_vaccine_tweets_with_sentiment.csv
[kaggle-dataset]: https://www.kaggle.com/datasets/datasciencetool/covid19-vaccine-tweets-with-sentiment-annotation
[gabriel-preda]: https://www.kaggle.com/gpreda
"""

# load dataset
import pandas as pd

df = pd.read_csv('./static/covid-19_vaccine_tweets_with_sentiment.csv', encoding='unicode_escape')

if st.checkbox('Show dataset head'):
    st.write('dataset head')
    st.write(df.head())

"""
## Analytics

To analyze what is the most common word that occur frequently,
we can use [CountVectorizer][count-vectorizer]. By using `CountVectorizer`,
we can get word occurences and its frequencies as shown in table below.
We can understand the table easily by creating a word cloud of it as shown
below too. 

[count-vectorizer]: https://scikit-learn.org/stable/modules/feature_extraction.html#common-vectorizer-usage
"""

# count frequencies using CountVectorizer
col1, col2 = st.columns([0.3, 0.7])

from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df['tweet_text'])
words = vectorizer.get_feature_names_out()
frequencies = X.toarray().sum(axis=0)

dfWords = pd.DataFrame({'word': words, 'frequency': frequencies})

col1.write(dfWords.sort_values('frequency', ascending=False).head())

# show cloud
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud
def showCloud(frequencies, col):
    radius = 500
    x, y = np.ogrid[:2 * radius, :2 * radius]

    mask = (x - radius) ** 2 + (y - radius) ** 2 > (radius - 20) ** 2
    mask = 255 * mask.astype(int)

    wc = WordCloud(background_color="white", repeat=True, mask=mask)
    wc.generate_from_frequencies(frequencies)

    fig, ax = plt.subplots()
    ax.axis("off")
    ax.imshow(wc, interpolation="bilinear")
    col.pyplot(fig)

showCloud(dict(zip(dfWords['word'], dfWords['frequency'])), col2)

"""
But, as we can see in the cloud, the biggest word is `the`,
which is meaningless. This indicate that the data needs to be cleaned.
One of the common cleaning methods is using [stopwords][stopwords].
By using stopwords, we could remove unimportant words, so that we can
easily see the interesting word as shown below.

[stopwords]: https://scikit-learn.org/stable/modules/feature_extraction.html#using-stop-words
"""

col1, col2 = st.columns([0.4, 0.6])

from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(stop_words='english')
X = vectorizer.fit_transform(df['tweet_text'])
words = vectorizer.get_feature_names_out()
frequencies = X.toarray().sum(axis=0)

dfWords = pd.DataFrame({'word': words, 'frequency': frequencies})

col1.write(dfWords.sort_values('frequency', ascending=False).head())

showCloud(dict(zip(dfWords['word'], dfWords['frequency'])), col2)
