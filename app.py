import requests
import streamlit as st
from streamlit_lottie import st_lottie

st.set_page_config(page_title="My WebPage", page_icon=":tada:", layout="wide")

def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()


lottie_depression = load_lottieurl("https://assets3.lottiefiles.com/packages/lf20_tijmpky4.json")


with st.container():
    st.subheader("Hi, we are Anaconda Team :wave:")
    st.title("A Depression Indicator")
    st.write("[Learn More> ](https://www.youtube.com/watch?v=4bBczzrciP0)")

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


