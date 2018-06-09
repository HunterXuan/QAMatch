import json
import jieba

from sklearn.feature_extraction.text import TfidfTransformer  
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import pysparnn.cluster_index as ci

import pickle

# path
# 数据保存路径
qa_path = './data/qa.json'
tv_path = './data/tv.pkl'
cp_path = './data/cp.pkl'

# qa = [
#	{
#		'q': 'question',
#		'a': 'answer'
#	}
# ]
qa = json.load(open(qa_path))

# Generate corpus
# 将 QA 连接起来并分词
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
# 词典
words = tv.get_feature_names()

# get feature
# 获取每对 QA 的TF-IDF
tfidf = tv.transform(corpus)

# build index
# 创建索引
cp = ci.MultiClusterIndex(tfidf, range(len(qa)))

# save
pickle.dump(tv, open(tv_path, 'wb'))
pickle.dump(cp, open(cp_path, 'wb'))