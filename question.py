from itertools import count
import streamlit as st
import numpy as np
import pandas as pd
import time
import requests

st.set_page_config(page_title="Beck's Depression Inventory", page_icon=":tada:", layout="wide")


with st.container():
    st.title("Beck's Depression Inventory")


q1 = st.radio(
"Question 1",
('I do not feel sad.',
'I feel sad',
'I am sad all the time and I cant snap out of it.', 
'I am so sad and unhappy that I cant stand it. '))
if q1 == 'I do not feel sad.':
    a = 0
elif q1 == 'I feel sad':
    a = 1
elif q1 == 'I am sad all the time and I cant snap out of it.': 
    a = 2
else:
    a = 3


q2 = st.radio(
"Question 2",
('I am not particularly discouraged about the future.',
'I feel discouraged about the future.',
'I feel I have nothing to look forward to.', 
'I feel the future is hopeless and that things cannot improve.'))
if q2 == 'I am not particularly discouraged about the future.':
    b = 0
elif q2 == 'I feel discouraged about the future.':
    b = 1
elif q2 == 'I feel I have nothing to look forward to.':
    b = 2
else: 
    b = 3

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

q4 = st.radio(
"Question 4",
('I get as much satisfaction out of things as I used to.',
'I don\'t enjoy things the way I used to',
'I don\'t get real satisfaction out of anything anymore. ', 
'I am dissatisfied or bored with everything. '))

if q4 == 'I get as much satisfaction out of things as I used to.':
    d = 0
elif q4 == 'I don\'t enjoy things the way I used to':
    d = 1
elif q4 == 'I don\'t get real satisfaction out of anything anymore. ':
    d = 2
else: 
    d = 3

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
elif q5 =='I feel quite guilty most of the time.':
    e = 2 
else:
    e = 3

q6 = st.radio(
"Question 6",
('I don\'t feel I am being punished. ',
'I feel I may be punished. ',
'I expect to be punished. ', 
'I feel I am being punished. '))

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

if q7 =='I don\'t feel disappointed in myself. ':
    g =0 
elif q7 == 'I am disappointed in myself.':
    g =1
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
elif q8 =='I blame myself all the time for my faults.':
    h =2
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
    j =3

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
    l =2
else: 
    l = 3

q13 = st.radio(
"Question 13",
('I make decisions about as well as I ever could.',
'I put off making decisions more than I used to. ',
'I have greater difficulty in making decisions more than I used to.', 
'I can\'t make decisions at all anymore.' ))

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
    o =2
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

with st.container():
    
    st.subheader("Level of Depression")
    st.write("1-10 These ups and downs are considered normal ")
    st.write("11-16 Mild mood disturbance")
    st.write("17-20 Borderline clinical depression")        
    st.write("21-30 Moderate depression")   
    st.write("31-40 Severe depression")   
    st.write("over 40 Extreme depression")   
    st.write("---") 

if st.button('Your Score'):
    st.success(a + b + c + d + e + f + g + h + i + j + k + l + m + n + o + p + q + r + s + t + u)

       
