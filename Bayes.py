from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
import pickle
import random
import os
# import config
import pandas as pd

history = pd.read_csv('data/Semantic/train_augmented.txt', sep='\t', encoding='utf8')
data = list(history.itertuples(index=False))
random.shuffle(data)
y, x = zip(*data)
x_train, x_test, y_train, y_test = train_test_split(x, y)
print(len(x_train))
print(len(y_train))
bow = CountVectorizer(analyzer='word', ngram_range=(1, 1))
tf_idf = TfidfVectorizer()
metrics = bow.fit_transform(x_train)
# metrics = tf_idf.fit_transform(x_train)
print(metrics.toarray().shape)
model = MultinomialNB()
model.fit(metrics, y_train)
print(model.predict(bow.transform(x_test[:1])))
print(model.score(bow.transform(x_test), y_test))
# print(model.score(tf_idf.transform(x_test), y_test))
sent = ['想 和你 聊天', '你好 啊', "今天天气 怎么样", "帮 我 查下 今天 的 股票"]
id = bow.transform(sent)
print(model.predict(id))
# with open("semantic.pkl", 'wb') as f:
#     pickle.dump(model, f)
with open("semantic.pkl", 'rb') as f:
    n_model = pickle.load(f)
# with open("bow.pkl", 'wb') as f:
#     pickle.dump(bow, f)
with open('bow.pkl', 'rb') as f:
    bow = pickle.load(f)

for x in sent:

    print("Sentence:", x, "Category:", n_model.predict(bow.transform([x])))


