import os
import sys
from ..logger import logging
from ..exception import CustomException
from ..utils import data_preprocessing
from ..utils import TokenPadding, save_object,model_trainer, train_model
from ..utils import SaveAndOpenTokenizer

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import keras
import joblib


#from keras.utils import pad_sequences 


class datatransformation():
    def __init__(self):
        self.preprocessor_file_path = os.path.join('archieve', 'encoder.pkl')
        self.train_arr_path = os.path.join('archieve', 'train_arr')
        self.test_arr_path = os.path.join('archieve', 'test_arr')


class MainTransform():
    def __init__(self):
        self.transformation = datatransformation()

            
    def data_transform_initiate(self,train_path, test_text_path):

        try:
            logging.info(f"Loading training data")
            
            df_train = pd.read_csv(train_path)

            logging.info(f"Loading testing data")

            df_test = pd.read_csv(test_text_path)

            logging.info(f"Reading necessary dataframes.")

            target_Column = 'Category'
            text_column = 'Text'

            input_data_text_features = df_train[[text_column]]
            input_data_target_features = df_train[[target_Column]]

            #Input labels are put in place

            target_data_text = df_test[[text_column]]
            df_label_data = df_test[[target_Column]]

            #Output data is put in place.

            ##Here we apply our first utils function to get data preprocessed and in a list

            logging.info("Preprocessing input text data.")

            input_text_preprocessed = data_preprocessing(df=input_data_text_features,col_name="Text")
            target_data_preprocessed = data_preprocessing(df=target_data_text,col_name='Text')

            ##Now we get them properly tokenized and padded and in numpy format.
            logging.info("Tokenizing and padding the input data.")
            num_words = 5000
            train_tokenizer = TokenPadding(num_words=num_words)
            X_train_final ,max_len, token1 = train_tokenizer.X_train_sequenced(input_text_preprocessed)           
            y_train_final = train_tokenizer.y_token_sequenced(target_data_preprocessed,max_len)
            
            logging.info(f'max_len value is {max_len}')

            logging.info("Tokenizing and padding successful")

            tokenizer_saver = SaveAndOpenTokenizer(token1)
            tokenizer_saver.SavingTokeinzer() 

            ##x_train_final and y_final_data prepared.

            logging.info("Tokenizing and padding of target (text) data complete.")

            ##Now we will deal with the target data
            
            logging.info("Encoding target labels.")
            target_encoder = OneHotEncoder(sparse_output=False)
            X_test_ = np.array(input_data_target_features).reshape(-1,1)
            y_test_ = np.array(df_label_data).reshape(-1,1)

            X_test_final = target_encoder.fit_transform(X_test_)
            y_test_final = target_encoder.transform(y_test_)

            #print(f'{np.shape(X_train_padded)},{np.shape(input_data_target_transformed)},{np.shape(y_train_padded)},{np.shape(df_label_data_transformed)}')   
            train_arr = np.c_[X_train_final, X_test_final]
            test_arr = np.c_[y_train_final,y_test_final]

            logging.info("Saving preprocessing object.")
            save_object(
                file_path = self.transformation.preprocessor_file_path,
                obj= target_encoder
            )

            np.save(self.transformation.train_arr_path,train_arr)
            np.save(self.transformation.test_arr_path, test_arr)
            
            logging.info("Data transformation complete.")

            logging.info("Initializing model training.")
            model, summary = model_trainer(input_dim=num_words + 1, input_length=max_len)

            model,history = train_model(model, X_train=X_train_final, 
                                X_test=X_test_final, 
                                y_train=y_train_final,
                                y_test=y_test_final, 
                                epochs=10, batch_size=32)

            ##here we can change a bit, the saving method.
            model.save(r'C:\Users\rgarlay\Desktop\DS\news_class\bbc_news_class\archieve\model.h5')

            train_acc = history.history['accuracy']
            val_acc = history.history['val_accuracy']
            
            logging.info(f"Training accuracy over epochs: {train_acc}")
            logging.info(f"Validation accuracy over epochs: {val_acc}")

            logging.info("Data transformation and model training complete.")

            return train_arr, test_arr, self.transformation.preprocessor_file_path
        
        except Exception as e:
            logging.error(f"Error during data transformation: {e}")
            raise CustomException(e,sys)
        
