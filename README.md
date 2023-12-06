# Conversational_AI_Chatbot
## Description
### The Chatbot is designed to classify the patients medical diagnosis based on his medical summary given by the physician.


## About the application
### The app is build using python as a programming language  and streamlit library for building an interactive UI

## Steps to start the app
### Python Version: Python 3.9.11
### Install all the required libraries using the following command
### pip install -r requirements

### Run the command: python model_test.py which will train and build a Text classifier using TFBertModel
### After successful execution of this script the below two pickle files will be created in the base folder 
### 1. tokenizer.pickle  -- (Already added the one created in the repo)
### 2. model.h5  -- Shared a drive link 

### Add the input files path and saved models files path in the config.ini
### Now run the strealit app using the command:
### streamlit run run_app.py

### This will run an app on the localhost which will show an interactive UI as shown below

<img width="623" alt="image" src="https://github.com/shrinalawade/Conversational_AI_Chatbot/assets/26817905/6883798f-9855-4d34-8cca-86bcec17f90f">

### The classfication metrics obtained after train-test split on the train data (80:20) as given below
<img width="306" alt="image" src="https://github.com/shrinalawade/Conversational_AI_Chatbot/assets/26817905/19f11fb0-9cb6-466f-8ba0-a1a19524636d">



