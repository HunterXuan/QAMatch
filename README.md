# QAMatch
A simple QA match method that combines bow and tf-idf

基于 BOW 和 TF-IDF 的简易 QA 匹配模型

# How to use | 使用方法
```bash
# 建立索引
python gen.py
# 查找匹配的问题
python search.py "第一个飞上太空的人是谁"
# 以下为程序输出内容
Building prefix dict from the default dictionary ...
Loading model from cache C:\Users\swxua\AppData\Local\Temp\jieba.cache
Loading model cost 0.791 seconds.
Prefix dict has been built succesfully.
Top matched QA:
=====================
Q:谁是第一个进入太空的人
A:推荐答案世界公认的第一位太空人是前苏联的尤里·阿列克谢耶维奇·加加林，他于1957年进入太空
=====================
```

# Extend QA Collection | 扩展 QA 库
```python
# build your own qa collection according to the following format
# 如果需要更换或者扩展 QA 库，修改 data 目录下的 qa.json 文件，格式如下
# [
#	  {
#		  'q': 'question',
#		  'a': 'answer'
#	  },
#   ...
#   {
#		  'q': 'question',
#		  'a': 'answer'
#	  }
# ]
```
