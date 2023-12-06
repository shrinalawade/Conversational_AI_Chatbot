import numpy as np
from input_data import input_data
from sklearn.model_selection import train_test_split
import tensorflow as tf
from transformers import AutoTokenizer,TFBertModel
from tensorflow.keras.optimizers.legacy import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Input, Dense
import pickle
from sklearn.metrics import classification_report

import logging
from logging.handlers import RotatingFileHandler
logger = logging.getLogger('Bert_Text_Classifier')
logger.setLevel(logging.DEBUG)
handler = RotatingFileHandler(r'./Bert_Text_Classifier.log',maxBytes=100000000, backupCount=5)
formatter = logging.Formatter('%(levelname)s %(asctime)s %(module)s %(process)d %(thread)d %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)


train, test = input_data()

train_df, val_df = train_test_split(train, test_size=0.2, random_state=100)


encoded_dict = { "digestive system diseases": 1,"cardiovascular diseases": 2, "neoplasms": 3,
     "nervous system diseases": 4, "general pathological conditions": 5}

y_train = to_categorical(train_df.category)
y_val = to_categorical(val_df.category)

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
bert = TFBertModel.from_pretrained("bert-base-cased")

# here tokenizer using from bert-base-cased
x_train = tokenizer(
    text=train_df.patient_summary.tolist(),
    add_special_tokens=True,
    max_length=150,
    truncation=True,
    padding=True,
    return_tensors='tf',
    return_token_type_ids = False,
    return_attention_mask = True,
    verbose = True)
x_val = tokenizer(
    text=val_df.patient_summary.tolist(),
    add_special_tokens=True,
    max_length=150,
    truncation=True,
    padding=True,
    return_tensors='tf',
    return_token_type_ids = False,
    return_attention_mask = True,
    verbose = True)


# Model Building
max_len = 150
input_ids = Input(shape=(max_len,), dtype=tf.int32, name="input_ids")
input_mask = Input(shape=(max_len,), dtype=tf.int32, name="attention_mask")
embeddings = bert(input_ids,attention_mask = input_mask)[0]
out = tf.keras.layers.GlobalMaxPool1D()(embeddings)
out = Dense(128, activation='relu')(out)
out = tf.keras.layers.Dropout(0.1)(out)
out = Dense(32,activation = 'relu')(out)
y = Dense(6,activation = 'softmax')(out)
model = tf.keras.Model(inputs=[input_ids, input_mask], outputs=y)
model.layers[2].trainable = True


optimizer = Adam(
    learning_rate=5e-05, # this learning rate is for bert model , taken from huggingface website
    epsilon=1e-08,
    decay=0.01,
    clipnorm=1.0)
model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])


checkpoint = tf.keras.callbacks.ModelCheckpoint('model.h5', monitor='val_accuracy', save_best_only=True, verbose=1)
earlystopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, verbose=1)


train_history = model.fit(
    x={'input_ids': x_train['input_ids'],'attention_mask': x_train['attention_mask']},
    y=y_train,
    validation_data=(
    {'input_ids': x_val['input_ids'],'attention_mask': x_val['attention_mask']}, y_val
    ),
    epochs=2,
    batch_size=36,
    callbacks=[checkpoint, earlystopping],
)

predicted_raw = model.predict({'input_ids':x_val['input_ids'],'attention_mask':x_val['attention_mask']})
logger.info(predicted_raw[0])


y_predicted = np.argmax(predicted_raw, axis = 1)
y_true = val_df.category

# Classification Report

logger.info(classification_report(y_true, y_predicted))

# Store the tokenizer object in a pickle

# saving
with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)


