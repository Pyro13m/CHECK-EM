import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

def analysis(sentence):
    df=pd.read_pickle("new_pkl.pkl")
    ps = PorterStemmer()
    corpus = []
    
    for i in range(1000):
        review = re.sub("[^a-zA-Z]"," ",df["Review"][i])
        review = review.lower()
        review = review.split()
        review = [ps.stem(word) for word in review if not word in set(stopwords.words("english"))]
        review = " ".join(review)
        corpus.append(review)
    
    review = re.sub("[^a-zA-Z]"," ",sentence)
    review = review.lower()
    review = review.split()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words("english"))]
    review = " ".join(review)
    corpus.append(review)   

    cv = CountVectorizer(max_features=1500)
    X = cv.fit_transform(corpus).toarray()
    y = df.iloc[:,1].values

    #print(X.shape)
    #print(y.shape)
    X_train = X[:-1,:]

    X_test = X[-1,:].reshape(1,-1)

    #X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.1,random_state=0)    
    classifier = GaussianNB()
    classifier.fit(X_train,y)
    
    #y_pred = classifier.predict(sentence)
    #y_pred = classifier.predict(X_test)
    
    #cm=confusion_matrix(y_test,y_pred)
    #sen_ =  cv.fit_transform().toarray()
    y_pred = classifier.predict(X_test)
    return y_pred