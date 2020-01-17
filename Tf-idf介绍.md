# TfIdf_Vector
_________
## 0 ```introduce Tfidf_Vec```

* [What is tf, idf and tfidf?](https://en.wikipedia.org/wiki/Tf%E2%80%93idf)  
* how to acquire tfidf value?    
 >> 1 acquire tf value of per word in document  
 >> 2 acquire idf value of per doc in documents  
 >> 3 acquire tf-idf vector of every word  
 
 ## 1 Tips of using gensim Tfidf_Vec<br/>
_________
### Steps:<br />
* 1 docs ---> dictionary  <br/>
> a. docs is a list consist of docutmnet <br/>
> b. use the ```jieba.cut(doc, cut_all=False)``` segmenting list<br/>
> c. use the gensim ```corpora.Dictionary(words)``` to construst a dictionary<br/>
> d. dictionary has many attributes(cfs,dfs,token2id...)<br/>
> e. use the new variable ```new_corpus= [dictionary.doc2bow(doc) for doc in docs]``` <br/>
* 2 dictionary --> gensim Tfidf model<br/>
> a. from gensim import models<br/>
> b. acquire tfidf training model```models.TfidfModel(new_corpus)```<br/>
> c. fun using the model to acquire vector of tfidf ``` tfidf_model[string_bow] ```<br/>
### *Friendly tips*:  
> for English sentence, gensim.TfIdfTranformer automaticlly rid of the stops words like "the, a, this..."  
> shape of ```dictionary.dow2bow(doc)``` is not same, in other words, beginning of calculating tfidf_vector, tfidf must be convert to the same dimension matrix.  
### Demo:
```Python  
from gensim import corpora
from gensim.sklearn_api import TfIdfTransformer  
words = [['红色', '的', '水果', '是', '什么'], [ '黎明', '中间', '是', '我'], ['我', '在', '小红', '和','黎明','之间'], ['是', '那里', '在', '我', '你']]  

 dictionary = corpora.Dictionary(words)  # dictionary存储了所有文档的单词内容，corpora.Dictionary()使word获得独立的id
 new_corpus = [dictionary.doc2bow(t) for t in words]
 
from gensim import models  
tfidf = models.TfidfModel(new_corpus)  # 作为模型  

# 获得某篇文档的tfidf值  
tfidf[new_corpus[0]]  
 ```
 ## 2 Python constuct TfIdf_vec
 ### Demo:
 
* tf_function
```Python
def custom_wlocal(x):

    ll = dict() # term frequency
    for i, m in Counter(x).items():
        ll[i] = Counter(x)[i] / np.sum(list(Counter(x).values()))
    return ll
```
* idf_function
```Python
from collections import defaultdict
from math import log10
from pandas import DataFrame
def custom_wglobal(x):
    # 遍历一遍，拿到所有的单词
    words = set()
    for n in x:
        for i in n:
            words.add(i)
    #针对每个不重复单词尽心文档的计数
    doc = defaultdict(float)
    try:
        for word in words:
            # 查找每个文档
            for n in x:
                if word in n:
                    doc[word] += 1
            #print(doc[word])
            doc[word] = log10(len(x) / (1.0 + doc[word]))
    except Exception as e:
        print(e)
    return doc
```
* tf_idf_value
```Python
    # 每篇文档的TF-IDF值d
def avg_tfidf(x, tf, idf):
    # 每篇文档tf
    Tf = [tf(tf_id) for tf_id in x]
    # 总idf
    Idf = idf(x)
    # 计算每篇的TF-IDF值
    text_of_tfidf = defaultdict(float)
    doc_of_tfidf = defaultdict(list)
    h = 0
    for doc in Tf:
        for i, m in doc.items():
            if i in Idf:
                text_of_tfidf[i] = doc[i] * Idf[i]
                
        h += 1
        doc_of_tfidf['doc'+ str(h)] = text_of_tfidf
        text_of_tfidf = defaultdict(float)
    return DataFrame(doc_of_tfidf).fillna(0)
```
* TfIdf_vector function
```Python
dataframe = avg_tfidf(x, custom_wlocal, custom_wglobal)  # 每一列就是tfidf向量
from math import sqrt
def vec_of_tfidf(dataframe):
    num = dataframe.shape[1]
    for i in range(num): 
        for j in range(1,num):
            fenzi = np.dot(dataframe.iloc[i], dataframe.iloc[j])
            fenmu = (sqrt(np.dot(dataframe.iloc[i],dataframe.iloc[i]))*sqrt(np.dot(dataframe.iloc[j],dataframe.iloc[j])))
        
        #cos_vec_tdidf = fenzi / fenmu
            print(i, i+1,fenzi/fenmu)
```
* 文本之间的相似度可以使用cos值进行衡量，但仍需要注意Tf-idf的几点：
>> * Tf-Idf是基于文本的，不是基于语义的，如果需要基于语义，还需要配合其他工具进行使用
>> * Tf-Idf针对较长的文本，某种层面上讲是包含语义关系的，一句话说的越长，包含的信息越精确（杠精请绕行！）
