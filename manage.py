

####Block
import pandas as pd
import re
import json
import nltk
import joblib
import pickle
import numpy as np
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score,classification_report

nltk.download('stopwords')
nltk.download('wordnet')


####Block
def loaddata():
    file_path="K:\\New folder (2)\\dataset.json"
    data=pd.read_json(file_path)
    return data
data=loaddata()
internal=data['internalStatus'].unique()
internal.sort()

####Block

def preprocessdata(data):

    ####Block
    data['externalStatus'] = data['externalStatus'].str.lower()
    data['internalStatus'] = data['internalStatus'].str.lower()

    ####Block
    data['externalStatus'] = data['externalStatus'].apply(lambda text: re.sub(r'[^a-zA-Z]', ' ', text))
    data['internalStatus'] = data['internalStatus'].apply(lambda text: re.sub(r'[^a-zA-Z]', ' ', text))

    ####Block
    data['externalStatus'] = data['externalStatus'].apply(lambda text: ' '.join(text.split()))  ####Block
    data['internalStatus'] = data['internalStatus'].apply(lambda text: ' '.join(text.split()))  ####Block

    ####Block
    stop_words = set(stopwords.words('english'))
    data['externalStatus'] = data['externalStatus'].apply(lambda text: ' '.join(word for word in text.split() if word not in stop_words))

    ####Block
    snowball_stemmer = SnowballStemmer("english")
    wordnet_lemmatizer = WordNetLemmatizer()

    data['externalStatus_stemmed'] = data['externalStatus'].apply(lambda text: ' '.join(snowball_stemmer.stem(word) for word in text.split()))
    data['externalStatus_lemmatized'] = data['externalStatus_stemmed'].apply(lambda text: ' '.join(wordnet_lemmatizer.lemmatize(word) for word in text.split()))
    data['internalStatus_stemmed'] = data['internalStatus'].apply(lambda text: ' '.join(snowball_stemmer.stem(word) for word in text.split()))
    data['internalStatus_lemmatized'] = data['internalStatus_stemmed'].apply(lambda text: ' '.join(wordnet_lemmatizer.lemmatize(word) for word in text.split()))
    ####Block
    data.dropna(inplace=True)

    return data

preprocessdata(data)

####Block
X = data['externalStatus_lemmatized']
y = data['internalStatus_lemmatized']
####Block
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

####Block
def split_data(X,y):
    X_train, X_test, y_train, y_test = train_test_split(X,y_encoded, test_size=0.2, random_state=42)
    return X_train,X_test,y_train,y_test

####Block
def vector(X_train):
    vectorizer = tf.keras.layers.TextVectorization(max_tokens=1000, output_mode='int')
    vectorizer.adapt(X_train.values)
    return vectorizer

####Block
def feedforward_model(vectorizer):
    model = tf.keras.Sequential([
        vectorizer,
        tf.keras.layers.Embedding(len(vectorizer.get_vocabulary()), 64, mask_zero=True),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(len(label_encoder.classes_), activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

####Block
def lstm_model(vectorizer):
    model_lstm = tf.keras.Sequential([
        vectorizer,
        tf.keras.layers.Embedding(len(vectorizer.get_vocabulary()), 64, mask_zero=True),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True)),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(len(label_encoder.classes_), activation='softmax')
    ])

    model_lstm.compile(optimizer='adam',
                       loss='sparse_categorical_crossentropy',
                       metrics=['accuracy'])
    return model_lstm

####Block
def random_forest(X,y):
    X_train, X_test, y_train, y_test =split_data(X,y)
    vectorizer = TfidfVectorizer(max_features=1000)

    X_train_tfidf = vectorizer.fit_transform(X_train)
    joblib.dump(vectorizer, "tfidf_vectorizer.joblib")

    X_test_tfidf = vectorizer.transform(X_test)
    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_classifier.fit(X_train_tfidf, y_train)

    y_pred = rf_classifier.predict(X_test_tfidf)

    rf_accuracy = accuracy_score(y_test, y_pred)
    print("Random Forest Test Accuracy:", rf_accuracy)
    return y_test,y_pred,rf_classifier


####Block
def train_feedforward(X,y):
    X_train, X_test, y_train, y_test = split_data(X,y)
    vectorizer=vector(X_train)
    model=feedforward_model(vectorizer)
    history=model.fit(X_train, y_train, epochs=10, validation_split=0.2)
    y_pred = model.predict(X_test)
    y_pred = np.argmax(y_pred, axis=1)
    loss, accuracy = model.evaluate(X_test, y_test)
    print("Feed forward Test Accuracy:", accuracy)
    return y_pred,y_test,model


####Block
def train_lstm(X,y):
    X_train, X_test, y_train, y_test = split_data(X,y)
    vectorizer=vector(X_train)
    model_lstm=lstm_model(vectorizer)
    ####Block
    history_lstm = model_lstm.fit(X_train, y_train, epochs=10, validation_split=0.2)
    y_pred = model_lstm.predict(X_test)
    y_pred = np.argmax(y_pred, axis=1)

    ####Block
    loss_lstm, accuracy_lstm = model_lstm.evaluate(X_test, y_test)
    print("Bidirectional LSTM Test Accuracy:", accuracy_lstm)
    return y_pred,y_test,model_lstm

####Block
def evaluate_measures_ff(X,y):
    y_pred_ff,y_test_ff,model=train_feedforward(X,y)
    report_ff = classification_report(y_test_ff, y_pred_ff)
    print("Classification Report Feedforward:\n", report_ff)
    return model

####Block
def evaluate_measures_lstm(X,y):
    y_pred_lstm,y_test_lstm,model_lstm=train_lstm(X,y)
    report_lstm = classification_report(y_test_lstm, y_pred_lstm)
    print("Classification Report LSTM:\n", report_lstm)

    return model_lstm

####Block
def evaluate_measures_rf(X,y):
    y_pred_rf,y_test_rf,model_rf=random_forest(X,y)
    report_rf = classification_report(y_test_rf, y_pred_rf)
    print("Classification Report Random forest:\n", report_rf)
    return model_rf


####Block
def ensemble_model(X,y):
    y_pred_ff,y_test_ff,model_ff=train_feedforward(X,y)
    y_pred_rf,y_test_rf,model_rf=random_forest(X,y)


    filename = "model_rf.joblib"
    joblib.dump(model_rf, filename)

    combined_pred = np.vstack((y_pred_rf,y_pred_ff)).T
    ensemble_pred = np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=1, arr=combined_pred)
    ensemble_accuracy = accuracy_score(y_test_ff, ensemble_pred)
    print("Ensemble Test Accuracy:", ensemble_accuracy)
    report_ensem = classification_report(y_test_ff,ensemble_pred)
    print("Classification Report Ensemble model:\n", report_ensem)
    return model_ff,model_rf

joblib.dump(ensemble_model, "ensemble_model.joblib")
model_ff,model_rf=ensemble_model(X,y)


evaluate_measures_lstm(X,y)
evaluate_measures_rf(X,y)