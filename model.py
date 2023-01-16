from pycaret.classification import load_model, predict_model
import streamlit as st
import pandas as pd
import numpy as np

def predict_quality(model, df):
    
    predictions_data = predict_model(estimator = model, data = df)
    return predictions_data['Label'][0]
    
model = load_model('lightgbmModelPersonality')


st.title('Personality Classification')
q1 = st.sidebar.slider('You regularly make new friends.', -1, 3)
q2 = st.sidebar.slider('You spend a lot of your free time exploring various random topics that pique your interest', -1, 3)
q3 = st.sidebar.slider('Seeing other people cry can easily make you feel like you want to cry too', -1, 3)
q4 = st.sidebar.slider('You often make a backup plan for a backup plan.', -1, 3)
q5 = st.sidebar.slider('You usually stay calm, even under a lot of pressure', -1, 3)
q6 = st.sidebar.slider('At social events, you rarely try to introduce yourself to new people and mostly talk to the ones you already know', -1, 3)
q7 = st.sidebar.slider('You prefer to completely finish one project before starting another.', -1, 3)
q8 = st.sidebar.slider('You are very sentimental.', -1, 3)
q9 = st.sidebar.slider('You like to use organizing tools like schedules and lists.', -1, 3)
q10 = st.sidebar.slider('Even a small mistake can cause you to doubt your overall abilities and knowledge.', -1, 3)
q11 = st.sidebar.slider('You feel comfortable just walking up to someone you find interesting and striking up a conversation.', -1, 3)
q12 = st.sidebar.slider('You are not too interested in discussing various interpretations and analyses of creative works.', -1, 3)
q13 = st.sidebar.slider('You are more inclined to follow your head than your heart.', -1, 3)
q14 = st.sidebar.slider('You usually prefer just doing what you feel like at any given moment instead of planning a particular daily routine.', -1, 3)
q15 = st.sidebar.slider('You rarely worry about whether you make a good impression on people you meet.', -1, 3)
q16 = st.sidebar.slider('You enjoy participating in group activities.', -1, 3)
q17 = st.sidebar.slider('You like books and movies that make you come up with your own interpretation of the ending.', -1, 3)
q18 = st.sidebar.slider('Your happiness comes more from helping others accomplish things than your own accomplishments.', -1, 3)
q19 = st.sidebar.slider('You are interested in so many things that you find it difficult to choose what to try next.', -1, 3)
q20 = st.sidebar.slider('You are prone to worrying that things will take a turn for the worse.', -1, 3)
q21 = st.sidebar.slider('You avoid leadership roles in group settings.', -1, 3)
q22 = st.sidebar.slider('You are definitely not an artistic type of person.', -1, 3)
q23 = st.sidebar.slider('You think the world would be a better place if people relied more on rationality and less on their feelings.', -1, 3)
q24 = st.sidebar.slider('You prefer to do your chores before allowing yourself to relax.', -1, 3)
q25 = st.sidebar.slider('You enjoy watching people argue.', -1, 3)
q26 = st.sidebar.slider('You tend to avoid drawing attention to yourself.', -1, 3)
q27 = st.sidebar.slider('Your mood can change very quickly.', -1, 3)
q28 = st.sidebar.slider('You lose patience with people who are not as efficient as you.', -1, 3)
q29 = st.sidebar.slider('You often end up doing things at the last possible moment.', -1, 3)
q30 = st.sidebar.slider('You have always been fascinated by the question of what, if anything, happens after death.', -1, 3)
q31 = st.sidebar.slider('You usually prefer to be around others rather than on your own.', -1, 3)
q32 = st.sidebar.slider('You become bored or lose interest when the discussion gets highly theoretical.', -1, 3)
q33 = st.sidebar.slider('You find it easy to empathize with a person whose experiences are very different from yours.', -1, 3)
q34 = st.sidebar.slider('You usually postpone finalizing decisions for as long as possible.', -1, 3)
q35 = st.sidebar.slider('You rarely second-guess the choices that you have made.', -1, 3)
q36 = st.sidebar.slider('After a long and exhausting week, a lively social event is just what you need.', -1, 3)
q37 = st.sidebar.slider('You enjoy going to art museums.', -1, 3)
q38 = st.sidebar.slider('You often have a hard time understanding other people’s feelings.', -1, 3)
q39 = st.sidebar.slider('You like to have a to-do list for each day.', -1, 3)
q40 = st.sidebar.slider('You rarely feel insecure.', -1, 3)
q41 = st.sidebar.slider('You avoid making phone calls.', -1, 3)
q42 = st.sidebar.slider('You often spend a lot of time trying to understand views that are very different from your own.', -1, 3)
q43 = st.sidebar.slider('In your social circle, you are often the one who contacts your friends and initiates activities.', -1, 3)
q44 = st.sidebar.slider('If your plans are interrupted, your top priority is to get back on track as soon as possible.', -1, 3)
q45 = st.sidebar.slider('You are still bothered by mistakes that you made a long time ago.', -1, 3)
q46 = st.sidebar.slider('You rarely contemplate the reasons for human existence or the meaning of life.', -1, 3)
q47 = st.sidebar.slider('Your emotions control you more than you control them.', -1, 3)
q48 = st.sidebar.slider('You take great care not to make people look bad, even when it is completely their fault.', -1, 3)
q49 = st.sidebar.slider('Your personal work style is closer to spontaneous bursts of energy than organized and consistent efforts.', -1, 3)
q50 = st.sidebar.slider('When someone thinks highly of you, you wonder how long it will take them to feel disappointed in you.', -1, 3)
q51 = st.sidebar.slider('You would love a job that requires you to work alone most of the time.', -1, 3)
q52 = st.sidebar.slider('You believe that pondering abstract philosophical questions is a waste of time.', -1, 3)
q53 = st.sidebar.slider('You feel more drawn to places with busy, bustling atmospheres than quiet, intimate places.', -1, 3)
q54 = st.sidebar.slider('You know at first glance how someone is feeling.', -1, 3)
q55 = st.sidebar.slider('You often feel overwhelmed.', -1, 3)
q56 = st.sidebar.slider('You complete things methodically without skipping over any steps.', -1, 3)
q57 = st.sidebar.slider('You are very intrigued by things labeled as controversial.', -1, 3)
q58 = st.sidebar.slider('You would pass along a good opportunity if you thought someone else needed it more.', -1, 3)
q59 = st.sidebar.slider('You struggle with deadlines.', -1, 3)
q60 = st.sidebar.slider('You feel confident that things will work out for you.', -1, 3)

data = [[q1,q2,q3,q4,q5,q6,q7,q8,q9,q10,q11,q12,q13,q14,q15,q16,q17,q18,q19,q20,q21,q22,q23,q24,q25,q26,q27,q28,q29,q30,
q31,q32,q33,q34,q35,q36,q37,q38,q39,q40,q41,q42,q43,q44,q45,q46,q47,q48,q49,q50,q51,q52,q53,q54,q55,q56,q57,q58,q59,q60]]
column = ['You regularly make new friends.', 'You spend a lot of your free time exploring various random topics that pique your interest', 
               'Seeing other people cry can easily make you feel like you want to cry too','You often make a backup plan for a backup plan.',
               'You usually stay calm, even under a lot of pressure','At social events, you rarely try to introduce yourself to new people and mostly talk to the ones you already know',
               'You prefer to completely finish one project before starting another.','You are very sentimental.','You like to use organizing tools like schedules and lists.',
               'Even a small mistake can cause you to doubt your overall abilities and knowledge.','You feel comfortable just walking up to someone you find interesting and striking up a conversation.',
               'You are not too interested in discussing various interpretations and analyses of creative works.','You are more inclined to follow your head than your heart.',
               'You usually prefer just doing what you feel like at any given moment instead of planning a particular daily routine.','You rarely worry about whether you make a good impression on people you meet.',
               'You enjoy participating in group activities.','You like books and movies that make you come up with your own interpretation of the ending.',
               'Your happiness comes more from helping others accomplish things than your own accomplishments.','You are interested in so many things that you find it difficult to choose what to try next.',
               'You are prone to worrying that things will take a turn for the worse.','You avoid leadership roles in group settings.','You are definitely not an artistic type of person.',
               'You think the world would be a better place if people relied more on rationality and less on their feelings.','You prefer to do your chores before allowing yourself to relax.',
               'You enjoy watching people argue.','You tend to avoid drawing attention to yourself.','Your mood can change very quickly.',
               'You lose patience with people who are not as efficient as you.','You often end up doing things at the last possible moment.','You have always been fascinated by the question of what, if anything, happens after death.',
               'You usually prefer to be around others rather than on your own.','You become bored or lose interest when the discussion gets highly theoretical.',
               'You find it easy to empathize with a person whose experiences are very different from yours.','You usually postpone finalizing decisions for as long as possible.',
               'You rarely second-guess the choices that you have made.','After a long and exhausting week, a lively social event is just what you need.',
               'You enjoy going to art museums.','You often have a hard time understanding other people’s feelings.','You like to have a to-do list for each day.',
               'You rarely feel insecure.','You avoid making phone calls.','You often spend a lot of time trying to understand views that are very different from your own.',
               'In your social circle, you are often the one who contacts your friends and initiates activities.','If your plans are interrupted, your top priority is to get back on track as soon as possible.',
               'You are still bothered by mistakes that you made a long time ago.','You rarely contemplate the reasons for human existence or the meaning of life.',
               'Your emotions control you more than you control them.','You take great care not to make people look bad, even when it is completely their fault.',
               'Your personal work style is closer to spontaneous bursts of energy than organized and consistent efforts.','When someone thinks highly of you, you wonder how long it will take them to feel disappointed in you.',
               'You would love a job that requires you to work alone most of the time.','You believe that pondering abstract philosophical questions is a waste of time.',
               'You feel more drawn to places with busy, bustling atmospheres than quiet, intimate places.','You know at first glance how someone is feeling.',
               'You often feel overwhelmed.','You complete things methodically without skipping over any steps.','You are very intrigued by things labeled as controversial.',
               'You would pass along a good opportunity if you thought someone else needed it more.','You struggle with deadlines.','You feel confident that things will work out for you.'
               ]
df = pd.DataFrame(data, columns=column)
st.table(df) 

if st.button('Predict'):
    
    prediction = predict_quality(model, df)
    
    st.write(' Based on feature values, your personality is '+ str(prediction))