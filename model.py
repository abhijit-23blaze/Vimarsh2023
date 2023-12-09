#Importing the required Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing the dataset of Sections and their descriptions from the the-bharatiya-nyaya-sanhita-2023 
dataset = pd.read_csv('Sections.csv')
x = dataset.iloc[: , -1].values

#Simplifying each description
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = []
for i in range(0, 344):
  review = re.sub('[^a-zA-Z]', ' ', x[i])
  review = review.lower()
  review = review.split()
  ps = PorterStemmer()
  all_stopwords = stopwords.words('english')
  all_stopwords.remove('not')
  review = [ps.stem(word) for word in review if not word in set(all_stopwords)]
  review = ' '.join(review)
  corpus.append(review)

#Converting the description of the dataset to vectors which are used for training
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 2000)
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, 0].values

#Training the model on the vectors
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X, y)

#Predicting the results of section for a FIR statement after simplifying it, in three different ways(to predict multiple sections)
corpus2 = []
t = 'robbery bribery theft rape gang rape murder house breaking half murder extortion dacoity'
review = re.sub('[^a-zA-Z]', ' ', t)
review = review.lower()
review = review.split()
ps = PorterStemmer()
all_stopwords = stopwords.words('english')
all_stopwords.remove('not')
review = [ps.stem(word) for word in review if not word in set(all_stopwords)]
review = ' '.join(review)
corpus2.append(review)
corpus3 = corpus2[0].split()

resultArray = []
originalString = corpus2[0]
while originalString:
    words = originalString.split()
    originalString = ' '.join(words[1:])
    resultArray.append(originalString)

resultArray.pop()

y_pred = classifier.predict(cv.transform(corpus2).toarray())
print(y_pred)

y_pred = classifier.predict(cv.transform(corpus3).toarray())

ypred = set(y_pred)
try :
    ypred.remove(46)
except:
    print('')

print(ypred)

try:
    y_pred = classifier.predict(cv.transform(resultArray).toarray())
    
    ypred = set(y_pred)
    try :
        ypred.remove(46)
    except:
        print('')
    
    print(ypred)
except:
    print('')
