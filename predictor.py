import nltk

from nltk.stem import WordNetLemmatizer

import json
import pickle

import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD
import random
from keras.models import load_model
import json
import random
from configuration import Configuration


class Predictor:
    lemmatizer = WordNetLemmatizer()
    error_threshold = 0.25
    answers = dict()

    def init(self):
        nltk.download('punkt')
        nltk.download('wordnet')
        return self

    def configure(self, config_dir='./config'):
        self.config = Configuration().load(config_dir).get_config()
        if 'answers' in self.config:
            for answer in self.config['answers']:
                if type(answer['intent']) is list:
                    for intent in answer['intent']:
                        self.answers[intent] = answer['responses']
                else:
                    self.answers[answer['intent']] = answer['responses']
        return self

    def load_model(self, model_dir='./model'):
        self.model = load_model(model_dir + '/chatbot_model.h5')
        self.words = pickle.load(open(model_dir + '/words.pkl', 'rb'))
        self.classes = pickle.load(open(model_dir + '/classes.pkl', 'rb'))
        return self

    def predict_top_response(self, msg):
        ints = self.predict_class(msg)
        res = None
        if len(ints) > 0:
            res = ints[0]
            intent = res['intent']
            if intent in self.answers:
                res['response'] = random.choice(self.answers[intent])
        return res

    def clean_up_sentence(self, sentence):
        sentence_words = nltk.word_tokenize(sentence)
        sentence_words = [self.lemmatizer.lemmatize(word.lower()) for word in sentence_words]
        return sentence_words

    def bow(self, sentence):
        sentence_words = self.clean_up_sentence(sentence)
        bag = [0] * len(self.words)
        for s in sentence_words:
            for i, w in enumerate(self.words):
                if w == s:
                    bag[i] = 1
        return (np.array(bag))

    def predict_class(self, sentence):
        p = self.bow(sentence)
        res = self.model.predict(np.array([p]))[0]
        results = [[i, r] for i, r in enumerate(res) if r > self.error_threshold]
        results.sort(key=lambda x: x[1], reverse=True)
        return_list = []
        for r in results:
            return_list.append({"intent": self.classes[r[0]], "probability": str(r[1])})
        return return_list
