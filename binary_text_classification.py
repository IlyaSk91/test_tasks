from re import split,sub,compile
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer,WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import TruncatedSVD

file=[]
for string in open('C:/temp/sentiment_labelled_sentences/imdb_labelled.txt','r'):
    file.append(string.lower())

def split_file(file):
    file_split=[]
    file_split=[split('\t',string,maxsplit=1) for string in file]
    return file_split

pattern=compile('[...,!?"-;\n\t]')

def clean_text(text,labels):
    text=[sub(pattern,'',string) for string in text]
    labels=[int(sub('\n','',number)) for number in labels]
    return text,labels

file_split=split_file(file)
text,labels=clean_text([file_split[i][0] for i in range(len(file_split))],[file_split[i][1] for i in range(len(file_split))])

def remove_stopwords(text):
    stop_words = set(stopwords.words('english'))
    removed_stopwords=[]
    for i in range(len(text)):
        removed_stopwords.append(' '.join([word for word in text[i].split() if not word in stop_words]))
    return removed_stopwords

clear_text=remove_stopwords(text)

len_sentence=[]
len_sentence.append([len(clear_text[i].split()) for i in range(len(clear_text))])
plt.hist(len_sentence)
plt.xlabel('Number of words in sentence')
plt.ylabel('Number of sentences')
plt.show()

def stemmer(clear_text):
    ps=PorterStemmer()
    stem_text=[]
    for i in range(len(clear_text)):
        stem_text.append(' '.join([ps.stem(w) for w in clear_text[i].split()]))
    return stem_text

stemmer_text=stemmer(clear_text)

def lemma(clear_text):
    lemmatizer = WordNetLemmatizer()
    lemma_text=[]
    for i in range(len(clear_text)):
        lemma_text.append(' '.join([lemmatizer.lemmatize(w) for w in clear_text[i].split()]))
    return lemma_text

lemmatizer_text=lemma(clear_text)


vectorizer=TfidfVectorizer(binary=False,ngram_range=(1,1))
X=vectorizer.fit_transform(clear_text)


X_train, X_test, y_train, y_test = train_test_split(X, labels, train_size = 0.6)

estimation={}
for c in np.linspace(0.01,1,5):
    clf=LogisticRegression(C=c).fit(X_train,y_train)
    estimation.update({c:accuracy_score(y_test,clf.predict(X_test))})
    
for key,value in estimation.items():
    if value==sorted(estimation.values(),reverse=True)[:1]:
        print('LG accuracy_score for c=',key,'{:.2%}'.format(value))
        
#val=sorted(estimation.values(),reverse=True)[:1][0]
c=dict((v,k) for k,v in estimation.items()).get(sorted(estimation.values(),reverse=True)[:1][0])
clf=LogisticRegression(C=c).fit(X_train,y_train)
feature_and_coeff={feature:coeff for feature,coeff in zip(vectorizer.get_feature_names(),clf.coef_[0])}

positive_words={}
for most_positive in sorted(feature_and_coeff.items(),key=lambda x: x[1],reverse=True)[:7]:  
    positive_words.update({most_positive[0]:most_positive[1]})
    
negative_words={}
for most_negative in sorted(feature_and_coeff.items(),key=lambda x: x[1],reverse=False)[:7]:  
    negative_words.update({most_negative[0]:most_negative[1]})
    
ax = plt.subplot(121)  
plt.barh(np.arange(7),list(positive_words.values()))
plt.yticks(np.arange(7), positive_words.keys(),rotation=0)
plt.ylabel('most positive words',fontsize=12)
plt.xlabel('coeff of LG',fontsize=12)

ax = plt.subplot(122)  
plt.barh(np.arange(7),list(negative_words.values()))
plt.gca().invert_xaxis()
plt.yticks(np.arange(7), negative_words.keys(),rotation=0)
plt.ylabel('most negative words',fontsize=12)
plt.xlabel('coeff of LG',fontsize=12)
plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.25,wspace=0.55)

plt.show() 

#pca=TruncatedSVD(n_components=2).fit_transform(X)
#plt.scatter(pca[:,0],pca[:,1],c=labels)
