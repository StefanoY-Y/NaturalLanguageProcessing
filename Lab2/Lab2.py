import gensim

#Заданная пара слов "тракт" и "долина"

model = gensim.models.KeyedVectors.load_word2vec_format('cbow.txt', binary=False)

positive = ["ущелье_NOUN", "дорога_NOUN"]
negative = ["лес_NOUN"]
top_words = model.most_similar(positive=positive, negative=negative, topn=10)

for word, score in top_words:
    print(f"{word}: {score}")