import os
import sys
import json

import pandas as pd
import numpy as np
import dill
from src.exception import CustomException

from nltk.corpus import stopwords
import re
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.text import tokenizer_from_json
import pickle
from nltk.stem import WordNetLemmatizer

from keras.utils import pad_sequences
from keras.layers import LSTM, Dense,Embedding, Dropout
from keras import Sequential
from keras.optimizers import Adam 
from tensorflow.keras.callbacks import EarlyStopping



def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)
    
    except Exception as e:
            raise CustomException(e,sys)
    
def load_object(file_path):

    try:
        with open(file_path,'rb') as file_obj:
            return dill.load(file_obj)
                
    except Exception as e:
        raise CustomException(e,sys)

def data_preprocessing(df,col_name):
    try:
        lemma = WordNetLemmatizer()       
        combined_stopwords = set(stopwords.words('english'))
        text_cleaned = []
        for i in range(len(df)):
            text_value = df[col_name][i]
            if not isinstance(text_value, str):
               text_value = str(text_value) if text_value is not None else ''
            text = re.sub('[^a-zA-Z0-9]', ' ', text_value)
            text = text.lower()
            text = text.split()
            text = [word for word in text if word not in combined_stopwords]
            text = [lemma.lemmatize(word) for word in text]
            cleaned_text = ' '.join(text)
            text_cleaned.append(cleaned_text)
            
    except Exception as e:
        raise CustomException(e,sys)

    return text_cleaned


class TokenPadding:
    def __init__(self,num_words):

        self.tokenizer = Tokenizer(num_words, oov_token='<OOV>')

    def X_train_sequenced(self, sequence):
        try:
            token = self.tokenizer.fit_on_texts(sequence)

            max_length = max(len(seq) for seq in sequence)

            cleaned_data = []
            for i in range(len(sequence)):
                sequences_encoded = self.tokenizer.texts_to_sequences([sequence[i]])
                cleaned_data.append(sequences_encoded[0])

            X_train_final = np.array(pad_sequences(cleaned_data, maxlen=max_length, padding='pre'))

        except Exception as e:
            raise CustomException(e,sys)
            
        return X_train_final, max_length, self.tokenizer

    def y_token_sequenced(self, sequence2 ,max_length):
        self.max_length = max_length
        try:
            cleaned_data2 = []
            for i in range(len(sequence2)):
                sequences_encoded2 = self.tokenizer.texts_to_sequences([sequence2[i]])
                cleaned_data2.append(sequences_encoded2[0])

            y_train_final = np.array(pad_sequences(cleaned_data2, maxlen=self.max_length, padding='pre'))

        except Exception as e:
            raise CustomException(e,sys)

        return y_train_final


class SaveAndOpenTokenizer:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def SavingTokeinzer(self):
        try:
            directory = r'C:/Users/rgarlay/Desktop/DS/news_class/bbc_news_class/archieve'
            
            file_path = os.path.join(directory, 'tokenizer.json')
            
            token = self.tokenizer.to_json()

            with open(file_path, 'w') as f:
                f.write(token)

        except Exception as e:
            raise CustomException(e,sys)
        
class  OpeningTokenizer:
        def __init__ (self,path_open):
            self.path_open = path_open
            pass
        def OpenTokenizer(self):
            try:
                with open(self.path_open, 'r') as f:
                    tokenizer_read = f.read()
                    new_tokenizer = tokenizer_from_json(tokenizer_read)

            except Exception as e:
                raise CustomException(e,sys)
        
            return new_tokenizer


def model_trainer(input_dim, input_length):
    model = Sequential()
    model.add(Embedding(input_dim=input_dim, input_length=input_length, output_dim=128))  # output_dim is set to 128
    model.add(LSTM(128, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(5, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    summary = model.summary()

    return model,summary

def train_model(model, X_train, X_test, y_train, y_test, epochs=5, batch_size=32):


    early_stopping = EarlyStopping(monitor='acc', patience=3, min_delta=0.01, mode='max' , restore_best_weights=True)
    
    history = model.fit(X_train, X_test, validation_data=(y_train, y_test),
                        epochs=epochs, batch_size=batch_size, callbacks=[early_stopping])
    
    return model, history




