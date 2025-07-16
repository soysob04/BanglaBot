import json
import numpy as np
import random
import nltk
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import SGD

nltk.download('punkt')
from nltk.tokenize import word_tokenize

with open("data/intents.json", encoding="utf-8") as file:
    data = json.load(file)

sentences = []
labels = []
classes = []

for intent in data["intents"]:
    for pattern in intent["patterns"]:
        sentences.append(pattern)
        labels.append(intent["tag"])
    if intent["tag"] not in classes:
        classes.append(intent["tag"])

words = []
for sentence in sentences:
    w = word_tokenize(sentence)
    words.extend(w)

words = sorted(set(words))
classes = sorted(set(classes))

training_data = []
output_data = []

encoder = LabelEncoder()
y = encoder.fit_transform(labels)

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(tokenizer=word_tokenize, vocabulary=words)
X = cv.fit_transform(sentences).toarray()

model = Sequential()
model.add(Dense(128, input_shape=(len(X[0]),), activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(64, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(len(classes), activation="softmax"))

sgd = SGD(learning_rate=0.01, momentum=0.9)
model.compile(loss="sparse_categorical_crossentropy", optimizer=sgd, metrics=["accuracy"])
model.fit(X, y, epochs=200, batch_size=5, verbose=1)

model.save("banglabot_model.h5")
import pickle
pickle.dump(words, open("words.pkl", "wb"))
pickle.dump(classes, open("classes.pkl", "wb"))
pickle.dump(encoder, open("encoder.pkl", "wb"))
