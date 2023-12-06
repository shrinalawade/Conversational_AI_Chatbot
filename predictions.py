import tensorflow as tf
import transformers
import pickle
import configparser
config = configparser.ConfigParser()
config.read("config.ini")
tokenizer_file_path = config['SAVED_MODELS']['TOKENIZER']
bert_classifier_file_path = config['SAVED_MODELS']['BERT_CLASSIFIER']


with open(tokenizer_file_path, 'rb') as handle:
    tokenizer = pickle.load(handle)

loaded_model = tf.keras.models.load_model(bert_classifier_file_path,
                                          custom_objects={"TFBertModel": transformers.TFBertModel})


encoded_dict = { "digestive system diseases": 1, "cardiovascular diseases": 2, "neoplasms": 3,
     "nervous system diseases": 4, "general pathological conditions": 5}


def get_prediction(texts):
    x_test = tokenizer(
        text=texts,
        add_special_tokens=True,
        max_length=150,
        truncation=True,
        padding='max_length',
        return_tensors='tf',
        return_token_type_ids=False,
        return_attention_mask=True,
        verbose=True)

    validation = loaded_model.predict({'input_ids': x_test['input_ids'], 'attention_mask': x_test['attention_mask']})
    res = {}
    for key, value in zip(encoded_dict.keys(), validation[0]):
        res.update({key: value})
    return res
