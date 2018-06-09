import json
import jieba

from sklearn.feature_extraction.text import TfidfTransformer  
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import pysparnn.cluster_index as ci

import pickle
import argparse

# path
# 数据保存路径
qa_path = './data/qa.json'
tv_path = './data/tv.pkl'
cp_path = './data/cp.pkl'

# parse arguments
# 解析入参
parser = argparse.ArgumentParser()
parser.add_argument('question', type=str, help='Type your Question')
args = parser.parse_args()

# get the question
# 获取用户的提问
question = args.question

# dived question into words
# 分词
cutted_qustion = jieba.cut(question)
cutted_qustion = ' '.join(cutted_qustion)

# retrieve qa, tv and cp built in gen.pysparnn
# 加载之前保存的数据
qa = json.load(open(qa_path))
tv = pickle.load(open(tv_path, 'rb'))
cp = pickle.load(open(cp_path, 'rb'))

# construct search data
# 构造搜索数据
search_data = [cutted_qustion]
search_tfidf = tv.transform(search_data)

# search from cp, k is the number of matched qa that you need
# 搜索数据，会获取到前 k 个匹配的 QA
result_array = cp.search(search_tfidf, k=1, k_clusters=2, return_distance=False)
result = result_array[0]

print("Top matched QA:")
print('=====================')
for id in result:
    print('Q:' + qa[int(id)]['q'])
    print('A:' + qa[int(id)]['a'])
    print('=====================')