#webapp modellel
#python -m streamlit run webapp_model.py
import numpy as np
import sklearn as sk
import pandas as pd
import tensorflow as tf 
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import streamlit as st

if 'model' not in st.session_state:
    try:
        st.session_state.model = joblib.load("random_forest.pkl")

        st.session_state.done = 1
    except FileNotFoundError:
        st.session_state.done = 0

if 'model2' not in st.session_state:
    try:
        st.session_state.model2 = load_model('my_model.keras')
    except FileNotFoundError:
        st.session_state.okok=False



data = pd.read_csv('test.csv')

    #'Online boarding', 'Class', 'Type of Travel', 'Inflight entertainment', 'Seat comfort', 'Cleanliness', 'Inflight wifi service', 'Baggage handling', 'Inflight service'
data = data.drop(['Unnamed: 0','id','Gender','Customer Type','Age','Flight Distance','Departure/Arrival time convenient','Ease of Online booking','Gate location','Food and drink','On-board service','Leg room service','Checkin service','Departure Delay in Minutes','Arrival Delay in Minutes'], axis=1)
data = data[data['Online boarding'] != 0]
data = data[data['Inflight entertainment'] != 0]
data = data[data['Seat comfort'] != 0]
data = data[data['Cleanliness'] != 0]
data = data[data['Inflight wifi service'] != 0]
data = data[data['Baggage handling'] != 0]
data = data[data['Inflight service'] != 0]



data['satisfaction'] = data['satisfaction'].map({'neutral or dissatisfied': 0, 'satisfied': 1})
data['Class'] = data['Class'].map({'Eco': 1, 'Eco Plus' :2,'Business': 0})
data['Type of Travel'] = data['Type of Travel'].map({'Personal Travel':1, 'Business travel':0})
y_test= data["satisfaction"]
X_test = data.drop('satisfaction', axis=1)






if st.session_state.done == 0:
    data = pd.read_csv('train.csv')

    #'Online boarding', 'Class', 'Type of Travel', 'Inflight entertainment', 'Seat comfort', 'Cleanliness', 'Inflight wifi service', 'Baggage handling', 'Inflight service'
    data = data.drop(['Unnamed: 0','id','Gender','Customer Type','Age','Flight Distance','Departure/Arrival time convenient','Ease of Online booking','Gate location','Food and drink','On-board service','Leg room service','Checkin service','Departure Delay in Minutes','Arrival Delay in Minutes'], axis=1)
    data = data[data['Online boarding'] != 0]
    data = data[data['Inflight entertainment'] != 0]
    data = data[data['Seat comfort'] != 0]
    data = data[data['Cleanliness'] != 0]
    data = data[data['Inflight wifi service'] != 0]
    data = data[data['Baggage handling'] != 0]
    data = data[data['Inflight service'] != 0]

    data['satisfaction'] = data['satisfaction'].map({'neutral or dissatisfied': 0, 'satisfied': 1})
    data['Class'] = data['Class'].map({'Eco': 1, 'Eco Plus' :2,'Business': 0})
    data['Type of Travel'] = data['Type of Travel'].map({'Personal Travel':1, 'Business travel':0})
    

    y= data['satisfaction']
    X = data.drop('satisfaction', axis=1)


    X_train, X_test, y_train, y_test= train_test_split(X,y, test_size=0.2, random_state=42)



    model = RandomForestClassifier(n_estimators=100,max_depth=20)
    st.session_state.model = model.fit(X_train, y_train)
    joblib.dump(st.session_state.model, "random_forest.pkl")
    st.session_state.done = 1

le_Class_mapping = {'Eco': 1, 'Eco Plus' :2,'Business': 0}
le_TypeOfTravel_mapping = {'Personal Travel':1, 'Business travel':0}



model = st.session_state.model
model2 = st.session_state.model2
y_pred = model.predict(X_test)

y_pred_prob2 = model.predict(X_test) 
y_pred2 = (y_pred_prob2 >= 0.5).astype(int)
prediction_dict = ['Neutral or dissatisfied','Satisfied']

personality_descriptions = {
    'Neutral or dissatisfied': 'The Passenger was not satisfied with the services of the airline.',
    'Satisfied': 'The Passenger was overall pleased with the services of the airline.'
    }


st.title("AIRLINE SATISFACTION webapp")

#'Online boarding', 'Class', 'Type of Travel', 'Inflight entertainment', 'Seat comfort', 'Cleanliness', 'Inflight wifi service', 'Baggage handling', 'Inflight service'

online_boarding = st.number_input('Online Boarding', min_value=1, max_value=5, value=5)
clas = st.selectbox('Class', ['Eco', 'Eco Plus','Business',])
type_of_travel = st.selectbox('Type of Travel', ['Personal Travel', 'Business travel'])
inflight_entertainment = st.number_input('Inflight entertainment', min_value=1, max_value=5, value=5)
seat_comfort = st.number_input('Seat comfort', min_value=1, max_value=5, value=5)
cleanliness = st.number_input('Cleanliness', min_value=1, max_value=5, value=5)
inflight_wifi_service = st.number_input('Inflight wifi service', min_value=1, max_value=5, value=5)
baggage_handling = st.number_input('Baggage handling', min_value=1, max_value=5, value=5)
inflight_service = st.number_input('Inflight service', min_value=1, max_value=5, value=5)

#kapott adat átalakítása
class_num = le_Class_mapping[clas]
type_of_travel_num = le_TypeOfTravel_mapping[type_of_travel]

df = {
    'Type of Travel': [type_of_travel_num],
    'Class': [class_num],
    'Inflight wifi service': [inflight_wifi_service],
    'Online boarding': [online_boarding],
    'Seat comfort': [seat_comfort],
    'Inflight entertainment': [inflight_entertainment],
    'Baggage handling': [baggage_handling],
    'Inflight service': [inflight_service],
    'Cleanliness': [cleanliness]
}
df = pd.DataFrame(df)



#ai gondolkodik és kitalálja hogy:
prediction = model.predict(df)
prediction_prob2 = model2.predict(df) 
if prediction_prob2 >= 0.5:
    prediction=1
else:
    prediction2=0
    
predicted_satisfaction = prediction_dict[prediction[0]]
predicted_satisfaction2 = prediction_dict[prediction2[0]]

st.write(f'RandomForest Prediction: {prediction_dict[prediction[0]]}')
accuracy = accuracy_score(y_test, y_pred)
st.write(f'Accuracy: {accuracy:.4f}')

st.write(f'DeepLearning Prediction: {predicted_satisfaction2}')
accuracy2 = accuracy_score(y_test, y_pred2)
st.write(f'Accuracy: {accuracy2:.4f}')

if st.button(f'More detailed description of {predicted_satisfaction}'):
    st.session_state.satisfaction_selected = predicted_satisfaction

if 'satisfaction_selected' in st.session_state:
    st.markdown("### Description:")
    selected_satisfaction = st.session_state.satisfaction_selected
    st.write(personality_descriptions[selected_satisfaction])

