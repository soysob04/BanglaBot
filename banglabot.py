import random
import json
import pickle
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from tensorflow.keras.models import load_model

model = load_model("banglabot_model.h5")
words = pickle.load(open("words.pkl", "rb"))
classes = pickle.load(open("classes.pkl", "rb"))
encoder = pickle.load(open("encoder.pkl", "rb"))

with open("data/intents.json", encoding="utf-8") as file:
    intents = json.load(file)

def clean_input(sentence):
    return word_tokenize(sentence)

def bow(sentence, words):
    sentence_words = clean_input(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence):
    p = bow(sentence, words)
    res = model.predict(np.array([p]))[0]
    index = np.argmax(res)
    tag = classes[index]
    return tag

def get_response(tag):
    for intent in intents["intents"]:
        if intent["tag"] == tag:
            return random.choice(intent["responses"])

print("BanglaBot চালু হয়েছে! কিছু বলো...")

while True:
    msg = input("তুমি: ")
    if msg.lower() in ["বিদায়", "exit", "quit"]:
        print("BanglaBot: বিদায়!")
        break
    tag = predict_class(msg)
    response = get_response(tag)
    print("BanglaBot:", response)
