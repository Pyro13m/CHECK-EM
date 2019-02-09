#import pandas to deal with the dataset
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
#from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
#from sklearn.metrics import confusion_matrix
import xx

def cleaning(sentence):
	new_corpus=[]
    review = re.sub("[^a-zA-Z]"," ",sentence)
    review = review.lower()
    review = review.split()
	review = [ps.stem(word) for word in review if not word in set(stopwords.words("english"))]
    review = " ".join(review)
	
	#cv = CountVectorizer(max_features=1500)
    #X = cv.fit_transform(corpus).toarray()

    corpus_sen = [0]*1500

    for i in review.split:
    	for j in range(len(corpus_sen)):
    		if i==


    #X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0)    
    classifier = GaussianNB()
    classifier.fit(X_train,y_train)
    
    #y_pred = classifier.predict(sentence)
    #y_pred = classifier.predict(X_test)
    
    #cm=confusion_matrix(y_test,y_pred)
    #y_pred = classifier.predict(sentence)
    #eturn y_pred