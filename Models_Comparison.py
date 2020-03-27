import pandas
import numpy
import string
import collections

from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk import pos_tag

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC, SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Perceptron
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier


datalen = 10000
common_words_len = int(datalen /5)
dataset = pandas.read_csv("Sentiment_Analysis_Dataset.csv", error_bad_lines=False, nrows=datalen)
#dataset.sample(frac =1)
x = dataset[["SentimentText"]]
y = dataset[["Sentiment"]]
porter = PorterStemmer()
stop_words = stopwords.words('english')
logistic_regression = LogisticRegression()
svm = LinearSVC()
svc = SVC()
trees = DecisionTreeClassifier()
naive_bayes = MultinomialNB()
random = RandomForestClassifier()
perceptron = Perceptron(penalty ='l2')
neural_network = MLPClassifier()
#print(x.head())

sentenceX = x.apply(lambda row: sent_tokenize(row["SentimentText"]), axis=1)
wordX = x.apply(lambda row: word_tokenize(row["SentimentText"]), axis=1)
target = [i for i in y["Sentiment"]]

wordX = [[word.lower() for word in xyz] for xyz in wordX]
wordX = [[word for word in xyz if word not in stop_words] for xyz in wordX]
wordX = [[word for word in xyz if word not in stop_words] for xyz in wordX]
wordX = [[word for word in xyz if word.isalpha()] for xyz in wordX]
wordX = [[porter.stem(word) for word in xyz] for xyz in wordX]
wordX = [pos_tag(xyz) for xyz in wordX]

tags = ['NNP']
wordX = [[word[0] for word in xyz if word[1] not in tags] for xyz in wordX]
newWordX = []
for idk in wordX:
    newWordX.append(' '.join(word for word in idk))
#print(newWordX)
#print(len(wordX))


#print(wordX[1])
temp = []
for i in wordX:
    for j in i:
        temp.append(j)

#print(len(temp))
#print(len(list(set(temp))))
set_temp = list(set(temp))
counts = collections.Counter(temp)
common_words = sorted(set_temp, key=lambda x: -counts[x])
common_words = common_words[:common_words_len]


my_encoder = [[0 for i in range(common_words_len)] for i in range(datalen)]
count = 0
for idk in wordX:
    for idc in idk:
        if idc in common_words:
            index = common_words.index(idc)
            my_encoder[count][index] = 1
    count +=1


X_train, X_test, y_train, y_test = train_test_split(my_encoder, target, test_size =0.2, random_state =42)

print("For top ", common_words_len, " common words:")

#LogisticRegression
logistic_regression.fit(X_train, y_train)
y_pred = logistic_regression.predict(X_test)
print("Logistic Regression:", accuracy_score(y_test, y_pred) *100)

#SVM
svm.fit(X_train, y_train)
y_pred = svm.predict(X_test)
print("SVM:", accuracy_score(y_test, y_pred) *100)

#Trees
trees.fit(X_train, y_train)
y_pred = trees.predict(X_test)
print("Decision Trees:", accuracy_score(y_test, y_pred) *100)

#Bayes
naive_bayes.fit(X_train, y_train)
y_pred = naive_bayes.predict(X_test)
print("Naive Bayes:", accuracy_score(y_test, y_pred) *100)

#RandomForest
random.fit(X_train, y_train)
y_pred = random.predict(X_test)
print("Random Forest:", accuracy_score(y_test, y_pred) *100)

#Perceptron
perceptron.fit(X_train, y_train)
y_pred = perceptron.predict(X_test)
print("Perceptron:", accuracy_score(y_test, y_pred) *100)

"""
#Neural Networks
neural_network.fit(X_train, y_train)
y_pred = neural_network.predict(X_test)
print("NN: AS:", accuracy_score(y_test, y_pred) *100)

#SVC
svc.fit(X_train, y_train)
y_pred = svc.predict(X_test)
print("SVC: AS:", accuracy_score(y_test, y_pred) *100)
"""
