import streamlit as st
from textblob import TextBlob
import pandas as pd
import altair as alt
import requests
from streamlit_lottie import st_lottie
import numpy as np
import cv2
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import model_from_json
# from tensorflow.keras.utils import img_to_array
from tensorflow.keras.preprocessing.image import img_to_array
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration, VideoProcessorBase, WebRtcMode
from itertools import count
import time
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from typing import Optional
import random
import json
import torch
from model import NeuralNet
from nltk_utils import bagofwords, tokenize
from questions import askMore


# default text for all pages
st.set_page_config(page_title="Depression Detection Website", page_icon=":tada:", layout="wide")

# Text Analysis detection
def convert_to_df(sentiment):
    sentiment_dict = {'polarity': sentiment.polarity, 'subjectivity': sentiment.subjectivity}
    sentiment_df = pd.DataFrame(sentiment_dict.items(), columns=['metric', 'value'])
    return sentiment_df

# analyzing key words and the type of text based on the 3 classifier types  and calcualting the polarity score
def analyze_keyword_sentiment(docx):
    analyzer = SentimentIntensityAnalyzer()
    pos_list = []
    neg_list = []
    neu_list = []
    for i in docx.split():
        res = analyzer.polarity_scores(i)['compound']
        if res > 0.1:
            pos_list.append(i)
            pos_list.append(res)

        elif res <= -0.1:
            neg_list.append(i)
            neg_list.append(res)
        else:
            neu_list.append(i)
            
    # displaying result

    result = {'positives': pos_list, 'negatives': neg_list, 'neutral': neu_list}
    return result

# Emotion-Cam Implementation

# load model
def load_model():
    with open("emotion_model1.json", "r") as json_file:
        loaded_model_json = json_file.read()
        classifier = tf.keras.models.model_from_json(loaded_model_json)
    classifier.load_weights("emotion_model1.h5")
    return classifier
# 5 labels for the model are loaded
emotion_dict = {0:'angry', 1:'happy',2:'neutral', 3:'sad',4:'surprised'}

# #loading jason and creating model
# json_file = open('emotion_model1.json','r')
# loaded_model_json = json_file.read()
# json_file.close()
# classifier = tf.keras.models.model_from_json(loaded_model_json)

# #loading weights into new model
# classifier.load_weights("emotion_model1.h5")

#loading face
try:
    #haarcascade classifier is being looked up
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    # face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
except Exception as e:
    st.write("Error loading cascade classifier!")
    st.write(e)

RTC_CONFIGURATION = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

class VideoTransformer(VideoTransformerBase):
    def recv(self,frame):
        img = frame.to_ndarray(format="bgr24")

        #image is being converted to grayscale
        # using detectMultiScale to detect multiple faces at the same time
        img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            image = img_gray, scaleFactor = 1.3, minNeighbors =5 )
        
        # Print number of faces detected
        print("Number of faces detected:", len(faces))
        
        # drawing a rectangle when face is detected
        for (x,y,w,h) in faces:
            cv2.rectangle(img=img, pt1=(x,y),pt2=(
                x+w,y+h), color =(0,255,0), thickness =2)

            roi_gray = img_gray[y:y+h,x:x+w]
            roi_gray= cv2.resize(roi_gray,(48,48), interpolation=cv2.INTER_AREA)

            if np.sum([roi_gray]) != 0 :
                roi = roi_gray.astype('float') / 255.0
                roi = img_to_array(roi)
                roi = np.expand_dims(roi, axis=0)
                prediction = classifier.predict(roi)[0]
                maxindex = int(np.argmax(prediction))
                finalout = emotion_dict[maxindex]
                output = str(finalout)
                
                # Print emotion prediction
                print("Emotion prediction:", output)

            label_position = (x,y)
            cv2.putText(img,output,label_position,cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)

        return img



def main():

    st.title("Depression Detection Website")
    # st.subheader("Streamlit Projects")
    # setting up the menu
    menu = ["Home","Text Analysis","Emotion-Cam","BDI Questionnaire","Chat Buddy"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Home":
        # loading a lottie animation 
        def load_lottieurl(url):
            r = requests.get(url)
            if r.status_code != 200:
                return None
            return r.json()

        lottie_depression = load_lottieurl("https://assets3.lottiefiles.com/packages/lf20_tijmpky4.json")

        with st.container():
            st.subheader("Hi, we are Anaconda Team :wave:")

            st.write(
                """
                Hello, nice to meet you, we are team Anaconda!!
                We are a group of five from the Faculty of Computer Science and Information Technology at the University of Malaya.
                This website is a product of our assignment for course code WIA1006 Machine Learning,
                we were assigned to develop a program to indicate depression. 
                We believe that everyone is perfectly imperfect and it is ok to not be okay,
                never feel burdened to talk about your feelings because you matter and you are loved 💙
                """
                )
            st.title("A Depression Indicator")
            st.write("[This is for you 💙 > ](https://www.youtube.com/watch?v=D9OOXCu5XMg)")

        # -- WHAT I DO ----
        with st.container():
            st.write("---")
            left_column, right_column = st.columns(2)
            with left_column:
                st.header("What is Depression")
                st.write("##")
                st.write(
                    """
                    Depression is a condition characterized by feelings of sadness, 
                    hopelessness, and often worthlessness, accompanied by both physical and mental symptoms. 
                    Depression can best be described as sadness that can take over your life 
                    and impact your daily activities, causing you to not function as you normally would.
                    """

                )

            with right_column:
                st_lottie(lottie_depression, height=300, key="depression")

    elif choice == "Text Analysis":

        st.subheader("Tell us about how you feel on a daily basis, try to write more than 2 sentences !! 🥰")
        # using nlp algorithm
        with st.form(key='nlpForm'):
            # getting user input
            raw_text = st.text_area("Write Here") 
            submit_button = st.form_submit_button(label='Analyze')

        # layout
        col1, col2 = st.columns(2)
        if submit_button:

            with col1:
                # using textblob to detect sentiment type
                st.info("Outcome")
                sentiment = TextBlob(raw_text).sentiment
                st.write(sentiment)

                # checking the polarity
                if sentiment.polarity > 0:
                    st.markdown("Emotion:: Positive :smiley: ")
                elif sentiment.polarity < 0:
                    st.markdown("Emotion:: Negative :☹️: ")
                else:
                    st.markdown("Emotion:: Neutral 😐 ")

                # Dataframe of the polarity and subjectivity
                result_df = convert_to_df(sentiment)
                st.dataframe(result_df)

                # Visualization of bar chart
                c = alt.Chart(result_df).mark_bar().encode(
                    x='metric',
                    y='value',
                    color='metric')
                st.altair_chart(c, use_container_width=True)

            with col2:
                st.info("Key Word Analysis")
                # classifying key words under positive, neutral and negative types
                keyword_sentiments = analyze_keyword_sentiment(raw_text)
                st.write(keyword_sentiments)

    elif choice == "Emotion-Cam":
        # displaying the webcam window 
        st.header("Live Emotion-Cam")
        st.write("Click on start to use webcam and detect your emotion")
        webrtc_streamer(key="example", video_processor_factory=VideoTransformer,)

    elif choice == "BDI Questionnaire":
        def load_lottieurl(url):
            r = requests.get(url)
            if r.status_code != 200:
                return None
            return r.json()

        lottie_img = load_lottieurl("https://assets4.lottiefiles.com/packages/lf20_kd5rzej5.json")
        with st.container():
            st.write("---")
            left_column, right_column = st.columns(2)
            with left_column:
                st.title("Beck's Depression Inventory")


            with right_column:
                st_lottie(lottie_img, height=300, key="depression")

        #user only able to select one option
        q1 = st.radio(
            "Question 1",
            ('I do not feel sad.',
             'I feel sad',
             'I am sad all the time and I cant snap out of it.',
             'I am so sad and unhappy that I cant stand it. '))
        
        #set values of every radio button to make the system able to count the score at end
        if q1 == 'I do not feel sad.':
            a = 0
        elif q1 == 'I feel sad':
            a = 1
        elif q1 == 'I am sad all the time and I cant snap out of it.':
            a = 2
        else:
            a = 3
            
        #user only able to select one option
        q2 = st.radio(
            "Question 2",
            ('I am not particularly discouraged about the future.',
             'I feel discouraged about the future.',
             'I feel I have nothing to look forward to.',
             'I feel the future is hopeless and that things cannot improve.'))
        
        #set values of every radio button to make the system able to count the score at end
        if q2 == 'I am not particularly discouraged about the future.':
            b = 0
        elif q2 == 'I feel discouraged about the future.':
            b = 1
        elif q2 == 'I feel I have nothing to look forward to.':
            b = 2
        else:
            b = 3

        #user only able to select one option
        q3 = st.radio(
            "Question 3",
            ('I do not feel like a failure.',
             'I feel I have failed more than the average person',
             'As I look back on my life, all I can see is a lot of failures. ',
             'I feel I am a complete failure as a person. '))

        if q3 == 'I do not feel like a failure.':
            c = 0
        elif q3 == 'I feel I have failed more than the average person':
            c = 1
        elif q3 == 'As I look back on my life, all I can see is a lot of failures. ':
            c = 2
        else:
            c = 3

        #user only able to select one option
        q4 = st.radio(
            "Question 4",
            ('I get as much satisfaction out of things as I used to.',
             'I don\'t enjoy things the way I used to',
             'I don\'t get real satisfaction out of anything anymore. ',
             'I am dissatisfied or bored with everything. '))

        #set values of every radio button to make the system able to count the score at end
        if q4 == 'I get as much satisfaction out of things as I used to.':
            d = 0
        elif q4 == 'I don\'t enjoy things the way I used to':
            d = 1
        elif q4 == 'I don\'t get real satisfaction out of anything anymore. ':
            d = 2
        else:
            d = 3

        #user only able to select one option
        q5 = st.radio(
            "Question 5",
            ('I don\'t feel particularly guilty',
             'I feel guilty a good part of the time',
             'I feel quite guilty most of the time.',
             'I feel guilty all of the time. '))

        if q5 == 'I don\'t feel particularly guilty':
            e = 0
        elif q5 == 'I feel guilty a good part of the time':
            e = 1
        elif q5 == 'I feel quite guilty most of the time.':
            e = 2
        else:
            e = 3

        #user only able to select one option
        q6 = st.radio(
            "Question 6",
            ('I don\'t feel I am being punished. ',
             'I feel I may be punished. ',
             'I expect to be punished. ',
             'I feel I am being punished. '))

        #set values of every radio button to make the system able to count the score at end
        if q6 == 'I don\'t feel I am being punished. ':
            f = 0
        elif q6 == 'I feel I may be punished. ':
            f = 1
        elif 'I expect to be punished. ':
            f = 2
        else:
            f = 3

        q7 = st.radio(
            "Question 7",
            ('I don\'t feel disappointed in myself. ',
             'I am disappointed in myself.',
             'I am disgusted with myself. ',
             'I hate myself. '))

        #set values of every radio button to make the system able to count the score at end
        if q7 == 'I don\'t feel disappointed in myself. ':
            g = 0
        elif q7 == 'I am disappointed in myself.':
            g = 1
        elif q7 == 'I am disgusted with myself. ':
            g = 2
        else:
            g = 3

        q8 = st.radio(
            "Question 8",
            ('I don\'t feel I am any worse than anybody else. ',
             'I am critical of myself for my weaknesses or mistakes',
             'I blame myself all the time for my faults.',
             'I blame myself for everything bad that happens.  '))

        if q8 == 'I don\'t feel I am any worse than anybody else. ':
            h = 0
        elif q8 == 'I am critical of myself for my weaknesses or mistakes':
            h = 1
        elif q8 == 'I blame myself all the time for my faults.':
            h = 2
        else:
            h = 3

        q9 = st.radio(
            "Question 9",
            ('I don\'t have any thoughts of killing myself',
             'I have thoughts of killing myself, but I would not carry them out. ',
             'I would like to kill myself. ',
             'I would kill myself if I had the chance.'))

        if q9 == 'I don\'t have any thoughts of killing myself':
            i = 0
        elif q9 == 'I have thoughts of killing myself, but I would not carry them out. ':
            i = 1
        elif 'I would like to kill myself. ':
            i = 2
        else:
            i = 3

        q10 = st.radio(
            "Question 10",
            ('I don\'t cry any more than usual. ',
             'I cry more now than I used to.',
             'I cry all the time now. ',
             'I used to be able to cry, but now I can\'t cry even though I want to. '))

        if q10 == 'I don\'t cry any more than usual. ':
            j = 0
        elif q10 == 'I cry more now than I used to.':
            j = 1
        elif q10 == 'I cry all the time now. ':
            j = 2
        else:
            j = 3

        q11 = st.radio(
            "Question 11",
            ('I am no more irritated by things than I ever was.',
             'I am slightly more irritated now than usual. ',
             'I am quite annoyed or irritated a good deal of the time. ',
             'I feel irritated all the time. '))

        if q11 == 'I am no more irritated by things than I ever was.':
            k = 0
        elif q1 == 'I am slightly more irritated now than usual. ':
            k = 1
        elif q11 == 'I am quite annoyed or irritated a good deal of the time. ':
            k = 2
        else:
            k = 3

        q12 = st.radio(
            "Question 12",
            ('I have not lost interest in other people. ',
             'I am less interested in other people than I used to be.',
             'I have lost most of my interest in other people',
             'I have lost all of my interest in other people.'))

        if q12 == 'I have not lost interest in other people. ':
            l = 0
        elif q12 == 'I am less interested in other people than I used to be.':
            l = 1
        elif q12 == 'I have lost most of my interest in other people':
            l = 2
        else:
            l = 3

        q13 = st.radio(
            "Question 13",
            ('I make decisions about as well as I ever could.',
             'I put off making decisions more than I used to. ',
             'I have greater difficulty in making decisions more than I used to.',
             'I can\'t make decisions at all anymore.'))

        if q13 == 'I make decisions about as well as I ever could.':
            m = 0
        elif q13 == 'I put off making decisions more than I used to. ':
            m = 1
        elif 'I have greater difficulty in making decisions more than I used to.':
            m = 2
        else:
            m = 3

        q14 = st.radio(
            "Question 14",
            ('I don\'t feel that I look any worse than I used to.',
             'I am worried that I am looking old or unattractive.',
             'I feel there are permanent changes in my appearance that make me look unattractive.',
             'I believe that I look ugly. '))

        if q14 == 'I don\'t feel that I look any worse than I used to.':
            n = 0
        elif q14 == 'I am worried that I am looking old or unattractive.':
            n = 1
        elif q14 == 'I feel there are permanent changes in my appearance that make me look unattractive.':
            n = 2
        else:
            n = 3

        q15 = st.radio(
            "Question 15",
            ('I can work about as well as before',
             'It takes an extra effort to get started at doing something. ',
             'I have to push myself very hard to do anything. ',
             'I can\'t do any work at all. '))

        if q15 == 'I can work about as well as before':
            o = 0
        elif q15 == 'It takes an extra effort to get started at doing something. ':
            o = 1
        elif q15 == 'I have to push myself very hard to do anything. ':
            o = 2
        else:
            o = 3

        q16 = st.radio(
            "Question 16",
            ('I can sleep as well as usual.',
             'I don\'t sleep as well as I used to. ',
             'I wake up 1-2 hours earlier than usual and find it hard to get back to sleep. ',
             'I wake up several hours earlier than I used to and cannot get back to sleep. '))

        if q16 == 'I can sleep as well as usual.':
            p = 0
        elif q16 == 'I don\'t sleep as well as I used to. ':
            p = 1
        elif q16 == 'I wake up 1-2 hours earlier than usual and find it hard to get back to sleep. ':
            p = 2
        else:
            p = 3

        q17 = st.radio(
            "Question 17",
            ('I don\'t get more tired than usual.',
             'I get tired more easily than I used to. ',
             'I get tired from doing almost anything. ',
             'I am too tired to do anything. '))

        if q17 == 'I don\'t get more tired than usual.':
            q = 0
        elif q17 == 'I get tired more easily than I used to. ':
            q = 1
        elif q17 == 'I get tired from doing almost anything. ':
            q = 2
        else:
            q = 3

        q18 = st.radio(
            "Question 18",
            ('My appetite is no worse than usual. ',
             'My appetite is not as good as it used to be. ',
             'My appetite is much worse now.',
             'I have no appetite at all anymore. '))

        if q18 == 'My appetite is no worse than usual. ':
            r = 0
        elif q18 == 'My appetite is not as good as it used to be. ':
            r = 1
        elif q18 == 'My appetite is much worse now.':
            r = 2
        else:
            r = 3

        q19 = st.radio(
            "Question 19",
            ('I haven\'t lost much weight, if any, lately. ',
             'I have lost more than five pounds. ',
             'I have lost more than ten pounds.',
             'I have lost more than fifteen pounds. '))

        if q19 == 'I haven\'t lost much weight, if any, lately. ':
            s = 0
        elif q19 == 'I have lost more than five pounds. ':
            s = 1
        elif q19 == 'I have lost more than ten pounds.':
            s = 2
        else:
            s = 3

        q20 = st.radio(
            "Question 20",
            ('I am no more worried about my health than usual.',
             'I am worried about physical problems like aches, pains, upset stomach, or constipation.',
             'I am very worried about physical problems and it\'s hard to think of much else. ',
             'I am so worried about my physical problems that I cannot think of anything else.'))

        if q20 == 'I am no more worried about my health than usual.':
            t = 0
        elif q20 == 'I am worried about physical problems like aches, pains, upset stomach, or constipation.':
            t = 1
        elif q20 == 'I am very worried about physical problems and it\'s hard to think of much else. ':
            t = 2
        else:
            t = 3

        q21 = st.radio(
            "Question 21",
            ('I have not noticed any recent change in my interest in sex. ',
             'I am less interested in sex than I used to be. ',
             'I have almost no interest in sex. ',
             'I have lost interest in sex completely. '))

        if q21 == 'I have not noticed any recent change in my interest in sex. ':
            u = 0
        elif q21 == 'I am less interested in sex than I used to be. ':
            u = 1
        elif q21 == 'I have almost no interest in sex. ':
            u = 2
        else:
            u = 3

        #display types of depression based on their score
        with st.container():
            st.subheader("Level of Depression")
            st.write("1-10 These ups and downs are considered normal ")
            st.write("11-16 Mild mood disturbance")
            st.write("17-20 Borderline clinical depression")
            st.write("21-30 Moderate depression")
            st.write("31-40 Severe depression")
            st.write("over 40 Extreme depression")
            st.write("---")

        #calculate total score
        sum = a + b + c + d + e + f + g + h + i + j + k + l + m + n + o + p + q + r + s + t + u
        
        
        with st.container():
            #the score only will display when user click the button
            if st.button('Your Score'):
                
                #display what types of depression they suffer based on their score and possible diagnosis/help
                st.success(sum)
                if sum <= 10:
                    st.subheader('These ups and downs are considered normal ')
                    st.write("[Cheer up! 🥳 ](https://www.youtube.com/watch?v=PppkNH3bKV4)")

                elif sum > 10 and sum <= 16:
                    st.subheader('Mild mood disturbance')
                    st.write("[Cheer up! 🥳 ](https://www.youtube.com/watch?v=PppkNH3bKV4)")

                elif sum > 16 and sum <= 20:
                    st.subheader('Borderline clinical depression')
                    st.write("[Don't Worry!](https://www.emedicinehealth.com/depression_health/article_em.htm)")

                elif sum > 20 and sum <= 30:
                    st.subheader('Moderate depression')
                    st.write("[Talk to Me ](https://www.justanswer.com/sip/mental-health?r=ppc|ga|1|||&JPKW=medical%20psychiatrist&JPDC=S&JPST=&JPAD=468261766532&JPMT=p&JPNW=g&JPAF=txt&JPRC=1&JPOP=&cmpid=11185346152&agid=108444346294&fiid=&tgtid=kwd-4428157412&ntw=g&dvc=c&r=ppc|ga|1|||&JPKW=medical%20psychiatrist&JPDC=S&JPST=&JPAD=468261766532&JPMT=p&JPNW=g&JPAF=txt&JPRC=1&JPCD=&JPOP=&cmpid=11185346152&agid=108444346294&fiid=&tgtid=kwd-4428157412&ntw=g&dvc=c&gclid=CjwKCAjw46CVBhB1EiwAgy6M4tVzXE4evcoZYZHV5W1e7Gz6bcSmqIB6xV8KxvvO5hKCNKWTjYx)")

                elif sum > 30 and sum <= 40:
                    st.subheader('Severe depression')
                    st.write("[Emotional Support ](https://www.befrienders.org.my/centre-in-malaysia)")

                else:
                    st.subheader('Extreme depression')
                    st.write("[Emotional Support ](https://www.befrienders.org.my/centre-in-malaysia)")

    elif choice == "Chat Buddy":
        # loading a lottie animation
        def load_lottieurl(url):
            r = requests.get(url)
            if r.status_code != 200:
                return None
            return r.json()

        lottie_img = load_lottieurl("https://assets9.lottiefiles.com/private_files/lf30_rXyqXj.json")
        with st.container():
            st.write("---")
            left_column, right_column = st.columns(2)
            with left_column:
                st.title("Welcome to ChatBot!")
                st.write("Hi, I am Anaconda, your depression indicator bot!(type 'quit' to exit)")

            with right_column:
                st_lottie(lottie_img, height=300, key="depression")


        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        #to get response for the chat
        with open('intents.json', 'r') as json_data:
            intents = json.load(json_data)
        
        #trained data set
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
        
        #for user input for chat
        reply = st.text_input("You: ", "")
        sentence = reply

        if sentence == "quit":
            st.write("Thank you for talking to me! have a nice day~")

        sentence = tokenize(sentence)
        X = bagofwords(sentence, all_words)
        X = X.reshape(1, X.shape[0])
        X = torch.from_numpy(X).to(device)

        output = model(X)
        _, predicted = torch.max(output, dim=1)

        tag = tags[predicted.item()]

        probs = torch.softmax(output, dim=1)
        prob = probs[0][predicted.item()]
        #to decide how to reply on the chat
        if prob.item() > 0.75:
            for intent in intents['intents']:
                if tag == intent["tag"]:
                    print()
                    st.write(f"{bot_name}: {random.choice(intent['responses'])}")
        else:
            st.write(f"{bot_name}: I do not understand...")




    else:
        pass


if __name__ == '__main__':
    main()
