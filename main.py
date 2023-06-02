import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt


"""
# VADER - **sentiment analytics**

## Objective
This interactive app is to illustrate the performance of VADER compared to manual annotation.

## Dataset
[The dataset][the-dataset] used is taken from [kaggle][kaggle-dataset]. Created by [Gabriel Preda][gabriel-preda].
The dataset is a collection of tweets related to Covid-19 vaccines with
manually annotated sentiments (negative, neutral, positive).
Negative sentiment is labeled as 1, neutral as 2, and positive as 3.

[the-dataset]: app/static/covid-19_vaccine_tweets_with_sentiment.csv
[kaggle-dataset]: https://www.kaggle.com/datasets/datasciencetool/covid19-vaccine-tweets-with-sentiment-annotation
[gabriel-preda]: https://www.kaggle.com/gpreda
"""

df = pd.read_csv('./static/covid-19_vaccine_tweets_with_sentiment.csv', encoding='unicode_escape')

if st.checkbox('Show dataset head'):
    st.write('dataset head')
    st.write(df.head())

"""
The dataset consist of 6000 tweets with 3 classes of sentiments. Each class is distributed as follows:
"""

col1, col2, _, _ = st.columns(4)
counts = df['label'].value_counts()
col1.write(counts)
labels = {
    'negative': 1,
    'neutral': 2,
    'positive': 3
}
col2.write(pd.Series(labels, name='label'))

"""
## Performance

One of the aspects that affect the performance of VADER is the mapping from compound score to sentiment labels.
In which, we have to map scores from the range of [-1, 1] to sentiments label [negative, neutral, positive].

We can see that by adjusting the threshold of neutral sentiment labels, it affects the classifications result as seen below.

"""

neutral = st.slider('neutral', min_value=0.0, max_value=1.0, step=0.01, label_visibility='hidden')

f"""
neutral threshold = ±{neutral}
"""

import nltk

nltk.download('vader_lexicon')

from nltk.sentiment.vader import SentimentIntensityAnalyzer

sid = SentimentIntensityAnalyzer()

with st.spinner(text="doing VADER calculation..."):
    df['scores'] = df['tweet_text'].apply(sid.polarity_scores)
    df['compound'] = df['scores'].apply(lambda score_dict: score_dict['compound'])
    df['comp_score'] = df['compound'].apply(lambda c: labels['positive'] if c > neutral else (labels['negative'] if c < -neutral else labels['neutral']))

col1, col2 = st.columns(2)
col1.write('Manual class counts')
col1.write(counts)

col2.write('VADER class counts')
col2.write(df['comp_score'].value_counts())

from sklearn.metrics import classification_report
from sklearn.metrics import ConfusionMatrixDisplay

f"""
```bash
{classification_report(df['label'], df['comp_score'], labels=[1, 2, 3])}
```
"""

"""
As we play with the neutral sentiment threshold, we can see
that the performance of VADER is not as good as manual annotation.
Generally, VADER has a higher recall but lower precision.
Given threshold of 0, the accuracy 0.46. When increased into 0.75, we can get accuracy of 0.62.
The accuracy is higher, but if we look at the confusion matrix, we can see that it only good at
predicting neutral sentiment.
But it is not good at predicting negative and positive sentiment.
"""

fig, ax = plt.subplots()
ConfusionMatrixDisplay.from_predictions(df['label'], df['comp_score'], display_labels=labels, cmap="Greens", ax=ax)

st.pyplot(fig)
