import streamlit as st

bot_name = "Anaconda"
ans = []

def countt(sentence):
    if sentence == "0":
        ans.append(0)
    elif sentence == "1":
        ans.append(1)
    elif sentence == "2":
        ans.append(2)
    elif sentence == "3":
        ans.append(3)

def count3(ans):
    i = 0
    total = 0
    while i < 4:
        str_count1 = ans.count(i)
        total = total + str_count1*i
        i += 1
    if total < 10:
        st.write("You are in normal condition. You need to calm down a little bit. Perhaps you can chat with me?")
    elif total in range(10,14):
        st.write("You are in mild condition. You are tired and need time for yourself. Perhaps you can chat with me?" )
    elif total in range(14,20):
        st.write("You are in moderate depression. You are tired and need a mental theraphy. Perhaps you can tell me your problem?")
    elif total in range(21,27):
        st.write("You are in severe depression. You have to take care of yourself first before others. Perhaps you can tell me your problem?")
    elif total > 27:
        st.write("You are in extreme severe depression. You have to take care of yourself first before others. Perhaps you can tell me your problem?")






def askMore():
    for resp in questions:
        st.success(f"{bot_name}:"+ resp +" Rate from 0 to 3")
        sentence = st.text_input("You: ")
        countt(sentence)
    count3(ans)




questions = ["Here are some questionss for you. Please read each statement and rate 0, 1, 2 or 3 which indicates how much the statement applied to you over the past week.\n\t\t There are no right or wrong answers. Do not spend too muchtime on any statement.\n\n Do you found it hard to gradually relax after doing something that has made you tired or worried? ",
             "Do you aware of dryness of your mouth?",
             "Can you experience any positive feeling at all?",
             "Do you experienced breathing difficulty? \n (e.g. excessively rapid breathing,breathlessness in the absence of physical exertion)",
             "Do you found it difficult to work up the initiative to do things?",
             "Do you tended to over-react to situations?",
             "Do you experienced trembling? (e.g. in the hands)",
             "Did you felt that you were using a lot of nervous energy?",
             "Do you worried about situations in which you might panic and make a fool of yourself?",
             "Did you felt that I had nothing to look forward to?",
             "Do you found yourself getting agitated?",
             "Do you found it difficult to relax?",
             "Did you felt down-hearted and blue?",
             "Did you intolerant of anything that kept you from getting on with what you were doing",
             "Did you felt you were close to panic?",
             "Did you unable to become enthusiastic about anything?",
             "Did you felt you were not worth much as a person?",
             "Did you felt that you were rather touchy?",
             "Did you was aware of the action of your heart in the absence of physical exertion? \n (e.g. sense of heart rate increase, heart missing a beat)",
             "Did you felt scared without any good reason?",
             "Did you felt that life was meaningless?"
             ]