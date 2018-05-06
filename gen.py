import json
import jieba

from sklearn.feature_extraction.text import TfidfTransformer  
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import pysparnn.cluster_index as ci

# qa = [
#	{
#		'q': 'question',
#		'a': 'answer'
#	}
# ]
qa = json.load(open(r'./data/qa.json'))

# Generate corpus
corpus = []
for id,item in enumerate(qa):
    tmp = item['q'] + item['a']
    tmp = jieba.cut(tmp)
    tmp = ' '.join(tmp)
    corpus.append(tmp)

# Generate bag of word
# TfidfVectorizer is a combination of CountVectorizer and TfidfTransformer
# Here we use TfidfVectorizer

tv = TfidfVectorizer()

# deal with corpus
tv.fit(corpus)

# get all words
words = tv.get_feature_names()
print(len(words))

# get feature 
tfidf = tv.transform(corpus)

cp = ci.MultiClusterIndex(tfidf, corpus)

# search
search_data = [
	'爱情 公寓 陈美嘉',
    '台湾 香港 沉没',
    '杨利伟 是 谁',
    '开水',
    '分手 大师'
]

search_tfidf = tv.transform(search_data)

print(cp.search(search_tfidf, k=1, k_clusters=2, return_distance=False))

#print(features_vec.getnnz())