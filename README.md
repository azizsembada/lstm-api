# Long Short-Term Memory (LSTM) API

# Fitur

- Dinamis Label - sistem bisa menjalankan file dengan jumlah label yang dinamis

## Teknologi

Teknologi yang digunakan pada service ini adalah :

- [Django] - Python Framework

- [MySQL] - Database

- [NLTK] - Library platform untuk membangun program python dengan data bahasa manusia

- [pandas] - data analysis and manipulation tool

- [scikit-learn] - Library untuk classification, regression and clustering algorithms

- [TensorFlow] - software library for dataflow and differentiable programming across a range of tasks

## Installation

Long Short-Term Memory (LSTM) API requires [Python] v3.6.x

langkah instalasi

1. clone repo :

```sh

$ git clone https://github.com/azizsembada/lstm-api.git

```

2. buka terminal pada repo program dan jalankan

```sh

$ Scripts\activate

```

3. Buat database MySQL dengan nama lstm_auth dan import file lstm_auth.sql yang ada di root directori

4. buka source code ke direktori src/textpreprocessing kemudian buka file settings.py dan cari code berikut

```sh

DATABASES = {

'default': {

'ENGINE': 'django.db.backends.mysql',

'NAME': 'lstm_auth',

'USER': 'root',

'PASSWORD':'',

'HOST':'localhost',

'PORT':'3306',

}

}

```

pastikan user, password, host dan port sesuai dengan server anda.

4. kembali ke direktori src kemudian jalankan

```sh

$ python manage.py runserver

```

jika proses berhasil akan tampil seperti ini pada terminal **Starting development server at http://127.0.0.1:8000/**

## Documentasi API

Untuk menggunakan service ini anda perlu mengakses 2 Endpoint berikut :

### Endpoint 1 Model : HTTP Request

Untuk membuat model dari dataset yang anda kirim :

```sh

POST http://127.0.0.1:8000/api/model

```

### Parameters

| Parameters          |          | Deskripsi                                                                                                                          |
| ------------------- | -------- | ---------------------------------------------------------------------------------------------------------------------------------- |
| dataset             | required | **dataset** adalah data yang akan dibuat modelnya (dataset dikirm dalam bentuk JSON, banyak tools online untuk covert xls to JSON) |
| access_token        | required | **access_token** adalah token yang digunakan agar dapat mengakses service                                                          |
| access_token_secret | required | **access_token_secret** adalah token yang digunakan agar dapat mengakses service                                                   |

### Result

| Parameters | Deskripsi                                                                         |
| ---------- | --------------------------------------------------------------------------------- |
| code       | **200** jika token valid **401** jika token tidak valid                           |
| title      | LSTM API                                                                          |
| status     | **success** jika text berhasil dipreprocessing **warning** jika token tidak valid |
| category   | label dari dataset                                                                |
| history    | nilai epoch                                                                       |
| y_act      | label_test                                                                        |
| y_pred     | niali prediksi dari model.predict()                                               |
| accuracy   | nilai accuracy berdasarkan train model (model.fit())                              |
| precision  | nilai precision berdasarkan train model (model.fit())                             |
| recall     | nilai recall berdasarkan train model (model.fit())                                |
| model      | data model                                                                        |

### Example JSON data

```sh

{
    "dataset": [
                {"konten":"contoh dataset satu","label":"contoh label satu"},
                {"konten":"contoh dataset dua","label":"contoh label dua"},
                {"konten":"contoh dataset tiga","label":"contoh label tiga"},
                {"konten":"contoh dataset .... sampai ukuran max file Json 100 MB","label":"contoh label satu"}
                ],
    "access_token": "access token anda",
    "access_token_secret": "access token secret anda"
}

```

### Endpoint 2 Predict : HTTP Request

Untuk membuat model dari dataset yang anda kirim :

```sh

POST http://127.0.0.1:8000/api/

```

### Parameters

| Parameters          |          | Deskripsi                                                                                                                          |
| ------------------- | -------- | ---------------------------------------------------------------------------------------------------------------------------------- |
| dataset             | required | **dataset** adalah data yang akan dibuat modelnya (dataset dikirm dalam bentuk JSON, banyak tools online untuk covert xls to JSON) |
| model               | required | **model** adalah data JSON yang didapat dari respon endpoint model                                                                 |
| content             | required | **content** adalah data yang akan di test                                                                                          |
| access_token        | required | **access_token** adalah token yang digunakan agar dapat mengakses service                                                          |
| access_token_secret | required | **access_token_secret** adalah token yang digunakan agar dapat mengakses service                                                   |

### Result

| Parameters | Deskripsi                                                                         |
| ---------- | --------------------------------------------------------------------------------- |
| code       | **200** jika token valid **401** jika token tidak valid                           |
| title      | LSTM API                                                                          |
| status     | **success** jika text berhasil dipreprocessing **warning** jika token tidak valid |
| status     | successt                                                                          |
| predict    | nilai prediksi                                                                    |

### Example JSON data

```sh

{
    "dataset": [
                {"konten":"contoh dataset satu","label":"contoh label satu"},
                {"konten":"contoh dataset dua","label":"contoh label dua"},
                {"konten":"contoh dataset tiga","label":"contoh label tiga"},
                {"konten":"contoh dataset .... sampai ukuran max file Json 100 MB","label":"contoh label satu"}
                ],
    "content": "saya ingin mengetes",
    "model" : {"class_name": "Sequential", "config": {"name": "sequential_2", "layers": [{"class_name": "Embedding", "config": {"name": "embedding_2", "trainable": true, "batch_input_shape": [null, 58], "dtype": "float32", "input_dim": 38966, "output_dim": 128, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": 6}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": 58}}, {"class_name": "LSTM", "config": {"name": "lstm_2", "trainable": true, "dtype": "float32", "return_sequences": false, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "units": 128, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": 6}}, "recurrent_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": 6}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}}, {"class_name": "Dense", "config": {"name": "dense_2", "trainable": true, "dtype": "float32", "units": 2, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": 6}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "keras_version": "2.3.1", "backend": "tensorflow"}
    "access_token": "access token anda",
    "access_token_secret": "access token secret anda"
}

```

### NOTE !

**Free Source code, Hell Yeah!**

please join to my circle [buruhkoding]

[django]: https://www.django-rest-framework.org/r
[mysql]: https://www.mysql.com/
[nltk]: https://www.nltk.org/
[pandas]: https://pandas.pydata.org/
[python]: python.org/downloads/release/python-3610/
[buruhkoding]: https://www.linkedin.com/in/abdullah-aziz-sembada-29730088/
[scikit-learn]: https://scikit-learn.org/stable/
[tensorflow]: https://www.tensorflow.org/
