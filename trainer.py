import nltk
from nltk.stem import WordNetLemmatizer
import json
import pickle
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD
import random
from pathlib import Path
import yaml
import os
from configuration import Configuration

class Trainer:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.words = []
        self.classes = []
        self.documents = []
        self.ignore_words = ['?', '!']

    def init(self):
        nltk.download('punkt')
        nltk.download('wordnet')
        return self

    def configure(self, configDir='./config'):
        self.config = Configuration().load(configDir).get_config()
        if 'questions' in self.config:
            for question in self.config['questions']:
                if 'patterns' in question:
                    for pattern in question['patterns']:
                        w = [self.lemmatizer.lemmatize(w.lower()) for w in nltk.word_tokenize(str(pattern)) if
                             w not in self.ignore_words]
                        self.words.extend(w)
                        intent = question['intent']
                        self.documents.append((w, intent))
                        if intent not in self.classes:
                            self.classes.append(intent)
        return self

    def train(self,modelDir='./model'):
        self.words = sorted(list(set(self.words)))
        self.classes = sorted(list(set(self.classes)))
        training = []
        output_empty = [0] * len(self.classes)
        for doc in self.documents:
            bag = []
            pattern_words = doc[0]
            pattern_words = [self.lemmatizer.lemmatize(word.lower()) for word in pattern_words]
            for w in self.words:
                bag.append(1) if w in pattern_words else bag.append(0)
            output_row = list(output_empty)
            output_row[self.classes.index(doc[1])] = 1
            training.append([bag, output_row])
        random.shuffle(training)
        training = np.array(training)
        train_x = list(training[:, 0])
        train_y = list(training[:, 1])
        # print("Training data created")

        model = Sequential()
        model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(len(train_y[0]), activation='softmax'))

        sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

        hist = model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)

        if not os.path.exists(modelDir):
            os.makedirs(modelDir)
        model.save(modelDir + '/chatbot_model.h5', hist)
        with open(modelDir + '/words.pkl', 'wb') as f:
            pickle.dump(self.words, f)
        with open(modelDir + '/classes.pkl', 'wb') as f:
            pickle.dump(self.classes, f)

        return self
