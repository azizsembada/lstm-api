import json
from rest_framework import generics, mixins
from rest_framework.views import APIView
from rest_framework.response import Response
from django.shortcuts import get_object_or_404

from .serializers import SembadaapiSerializer
from sembadaapi.models import Sembadaapi
from django.core import serializers

import numpy as np
import pandas as pd
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from keras.models import model_from_json
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

import tensorflow as tf


def is_json(json_data):
    try:
        real_json = json.loads(json_data)
        is_valid = True
    except ValueError:
        is_valid = False
    return is_valid


class DetectWidgetAPIView(APIView):
    permission_classes = []
    authentication_classes = []

    serializer_class = SembadaapiSerializer

    def post(self, request, *args, **kwargs):
        post_data = json.loads(request.body)
        data = {}
        obj = Sembadaapi.objects.filter(access_token__exact=post_data["access_token"]) & Sembadaapi.objects.filter(
            access_token_secret__exact=post_data["access_token_secret"])
        if obj:
            train = pd.DataFrame(post_data["dataset"])
            train.replace('', np.nan, inplace=True)
            # memastikan tidak ada nilai yang kosong
            train.dropna(inplace=True)

            tokenizer = Tokenizer()
            tokenizer.fit_on_texts(train['konten'].values)
            maxlen = max([len(i.split()) for i in train['konten'].values])

            model = tf.keras.models.model_from_json(
                post_data["model"], custom_objects=None
            )
            kalimat = post_data["content"]
            kalimat = kalimat.lower()
            kalimat = sent_tokenize(kalimat)
            kalimat = tokenizer.texts_to_sequences(kalimat)
            kalimat = pad_sequences(kalimat, maxlen=maxlen)
            pred = model.predict(kalimat)
            data['code'] = 200
            data['title'] = 'LSTM API'
            data['text'] = 'Berhasil di analisis'
            data['status'] = 'success'
            data['predict'] = pred[0]
            return Response(data)
        else:
            data['code'] = 401
            data['title'] = 'LSTM API'
            data['text'] = 'Token akses tidak valid'
            data['status'] = 'warning'
            data['result'] = post_data["content"]

            return Response(data)
