import pandas
import numpy
import string
import collections
import re
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk import pos_tag

from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB

def hashtag_extract(given_list):
    hashtags = []
    for word in given_list:
        ht = re.findall(r"(\w+)#", word)
        hashtags.extend(ht)
    return hashtags


datalen = 40000
#common_words_len = int(datalen /100)
dataset = pandas.read_csv("Sentiment_Analysis_Dataset.csv", error_bad_lines=False, nrows=datalen)
#dataset.sample(frac =1)
x = dataset[["SentimentText"]]
y = dataset[["Sentiment"]]
porter = PorterStemmer()
stop_words = stopwords.words('english')
logistic_regression = LogisticRegression()
naive_bayes = MultinomialNB()

#print(x.head())

sentenceX = x.apply(lambda row: sent_tokenize(row["SentimentText"]), axis=1)
wordX = x.apply(lambda row: word_tokenize(row["SentimentText"]), axis=1)
wordY = x.apply(lambda row: row["SentimentText"].split(), axis=1)
target = [i for i in y["Sentiment"]]

wordX = [[word.lower() for word in xyz] for xyz in wordX]
wordX = [[word for word in xyz if word not in stop_words] for xyz in wordX]

hashtags = [hashtag_extract(xyz) for xyz in wordY]

wordX = [[word for word in xyz if word.isalpha()] for xyz in wordX]
for i in range(len(wordX)):
    wordX[i].extend(hashtags[i])

wordX = [[word for word in xyz if len(word) >3] for xyz in wordX]
wordX = [[porter.stem(word) for word in xyz] for xyz in wordX]
wordY = [pos_tag(xyz) for xyz in wordX]

#tags = ['NNP', 'NNPS', 'FW', 'PDT']
noun_tag = ['NN', 'PRP', 'PRP$']
verb_tag = ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']
adverb_tag = ['RB', 'RBR', 'RBS', 'WRB']
adjective_tag = ['JJ', 'JJR', 'JJS']
tags = {noun_tag:1, verb_tag:2, adverb_tag:3, adjective_tag:4}

#wordX = [[word[0] for word in xyz if word[1] not in tags] for xyz in wordY]

#print(wordX[1])
"""temp = []
for i in wordX:
    for j in i:
        temp.append(j)"""

#print(datalen)
#print(len(temp))
#print(len(list(set(temp))))
"""set_temp = list(set(temp))
counts = collections.Counter(temp)
common_words = sorted(set_temp, key=lambda x: -counts[x])
common_words = common_words[:common_words_len]"""

sia = SentimentIntensityAnalyzer()
"""for word in test_subset:
    if (sia.polarity_scores(word)['compound']) >= 0.25:
        pos_word_list.append(word)
    elif (sia.polarity_scores(word)['compound']) <= -0.25:
        neg_word_list.append(word)"""

my_encoder = []
for tweet in wordY:
    value = 0
    for word in tweet:
        tag = ''
        for x in tags:
            if word[1] in x:
                tag = x
        scale = tags.get(tag)
        if (sia.polarity_scores(word[0])['compound']) >= 0.25:
            value += scale
        elif (sia.polarity_scores(word[0])['compound']) <= -0.25:
            value -= scale
    my_encoder.append(value)

my_encoder = numpy.array(my_encoder).reshape(-1, 1)

X_train, X_test, y_train, y_test = train_test_split(my_encoder, target, test_size =0.2, random_state =42)

#print("\nFor top ", common_words_len, " common words:\n")

#LogisticRegression
logistic_regression.fit(X_train, y_train)
y_pred = logistic_regression.predict(X_test)
print("Logistic Regression F1_Score:", f1_score(y_test, y_pred))
print("Logistic Regression Accuracy:", accuracy_score(y_test, y_pred))

print('\n')

#Bayes
naive_bayes.fit(X_train, y_train)
y_pred = naive_bayes.predict(X_test)
print("Naive Bayes F1_Score:", f1_score(y_test, y_pred) *100)
print("Naive Bayes Accuracy:", accuracy_score(y_test, y_pred) *100)
