from typing import Optional

import streamlit as st
import random
import json
import torch
from model import NeuralNet
from nltk_utils import bagofwords, tokenize
from questions import askMore

st.set_page_config(page_title="ChatBot", page_icon=":speech_balloon:", layout="centered")

with st.container():
    st.title("Welcome to ChatBot!")
    st.write("Hi, I am Anaconda, your depression indicator bot!(type 'quit' to exit)")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('intents.json', 'r') as json_data:
    intents = json.load(json_data)

FILE = "data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = "Anaconda"

reply = st.text_input("You: ", "")
sentence = reply


#sentence = input("You: ")

if sentence == "quit":
    write("Thank you for talking to me! have a nice day~")
    #break

#if sentence == "yes":
    #askMore()
    #break

sentence = tokenize(sentence)
X = bagofwords(sentence, all_words)
X = X.reshape(1, X.shape[0])
X = torch.from_numpy(X).to(device)

output = model(X)
_, predicted = torch.max(output, dim=1)

tag = tags[predicted.item()]

probs = torch.softmax(output, dim=1)
prob = probs[0][predicted.item()]
if prob.item() > 0.75:
    for intent in intents['intents']:
        if tag == intent["tag"]:
            print()
            st.write(f"{bot_name}: {random.choice(intent['responses'])}")
else:
    st.write(f"{bot_name}: I do not understand...")


