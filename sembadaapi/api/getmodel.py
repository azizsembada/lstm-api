from .serializers import SembadaapiSerializer
from sembadaapi.models import Sembadaapi
from django.core import serializers

import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix
from nltk.tokenize import sent_tokenize
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split
import json
from rest_framework import generics, mixins
from rest_framework.views import APIView
from rest_framework.response import Response
from django.shortcuts import get_object_or_404

import numpy as np
import pandas as pd
from keras import initializers
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential, model_from_json
from keras.layers import Dense, Embedding, Dropout, LSTM


class GetModelAPIView(APIView):
    permission_classes = []
    authentication_classes = []

    serializer_class = SembadaapiSerializer

    def post(self, request, *args, **kwargs):
        post_data = json.loads(request.body)
        data_response = {}
        obj = Sembadaapi.objects.filter(access_token__exact=post_data["access_token"]) & Sembadaapi.objects.filter(
            access_token_secret__exact=post_data["access_token_secret"])
        if obj:
            data = pd.DataFrame(post_data["dataset"])
            data.replace('', np.nan, inplace=True)
            data.dropna(inplace=True)  # memastikan tidak ada nilai yang kosong

            # mengubah kategori menjadi angka
            kategori = data['label'].unique()
            for i in range(len(kategori)):
                data['label'].replace(
                    to_replace=kategori[i], value=i, inplace=True)

            text = data['konten'].values
            label = data['label'].values
            data_train, data_test, label_train, label_test = train_test_split(
                text, label, test_size=0.5, random_state=42)
            data_test, data_val, label_test, label_val = train_test_split(
                data_test, label_test, test_size=0.5, random_state=42)

            # Tokenizer
            tokenizer = Tokenizer()
            tokenizer.fit_on_texts(text)

            vocab = max([len(tokenizer.word_index)]) + 1  # kamus kata
            maxlen = max([len(i.split())
                          for i in text])  # panjang input sequence
            batch_size = 128  # penentuan jumlah sample yang ditraining pada tiap epoch
            num_epochs = 50  # banyak iterasi pada saat training model
            initializer = initializers.RandomUniform(
                minval=-0.05, maxval=0.05, seed=2)

            # Data Train
            X_train = tokenizer.texts_to_sequences(data_train)
            X_train = pad_sequences(X_train, maxlen=maxlen)
            Y_train = to_categorical(label_train, num_classes=len(kategori))

            # Data Test
            X_test = tokenizer.texts_to_sequences(data_test)
            X_test = pad_sequences(X_test, maxlen=maxlen)

            # Data Validation
            X_val = tokenizer.texts_to_sequences(data_val)
            X_val = pad_sequences(X_val, maxlen=maxlen)
            Y_val = to_categorical(label_val, num_classes=len(kategori))

            def get_model(X, Y):
                model = Sequential()
                model.add(Embedding(input_dim=vocab, output_dim=128,
                                    input_length=maxlen, embeddings_initializer=initializer))
                model.add(LSTM(128, recurrent_initializer=initializer,
                               kernel_initializer=initializer))
                model.add(Dense(len(kategori), activation='sigmoid',
                                kernel_initializer=initializer))
                model.compile(loss='binary_crossentropy',
                              optimizer='adam', metrics=['accuracy'])
                print(model.summary())

                return model

            def evaluation(model, X, Y):
                # Predict the values
                Y_pred = model.predict(X)
                Y_pred_class = np.argmax(Y_pred, axis=1)
                Y_act = Y
                # accuracy
                accuracy = accuracy_score(Y_act, Y_pred_class)
                print("Accuracy: %.2f" % (accuracy*100), "%")
                # precision
                precision = precision_score(
                    Y_act, Y_pred_class, average='micro')
                print("Precision: %.2f" % (precision*100), "%")
                # recall
                recall = recall_score(Y_act, Y_pred_class, average='micro')
                print("Recall: %.2f" % (recall*100), "%")

                return Y_act, Y_pred_class, accuracy, precision, recall

            # get model
            model = get_model(X_train, Y_train)

            # training model
            history = model.fit(X_train, Y_train, batch_size=batch_size,
                                epochs=num_epochs, verbose=1, validation_data=(X_val, Y_val))

            Y_act, Y_pred, accuracy, precision, recall = evaluation(
                model, X_test, label_test)

            model_json = model.to_json()

            data_response['code'] = 200
            data_response['title'] = 'LSTM API'
            data_response['text'] = 'Berhasil di analisis'
            data_response['status'] = 'success'
            data_response['category'] = kategori
            data_response['history'] = history.history
            data_response['y_act'] = Y_act
            data_response['y_pred'] = Y_pred
            data_response['accuracy'] = accuracy
            data_response['precision'] = precision
            data_response['recall'] = recall
            data_response['model'] = model_json
            return Response(data_response)
        else:
            data_response['code'] = 401
            data_response['title'] = 'LSTM API'
            data_response['text'] = 'Token akses tidak valid'
            data_response['status'] = 'warning'
            data_response['result'] = ""

            return Response(data_response)
