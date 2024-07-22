import streamlit as st
import pickle 
import numpy as np   
data=pickle.load(open("C:/Users/Delip/2111CS010124/Diabetes.sav", "rb"))
# def load_model():
#     with open("C:/Users/Delip/2111CS010124/Diabetes.pkl", "rb") as file:
#         data = pickle.load(file)
#     return data

# data=load_model()

st.title("""RESTAURANT RATING REDECTION""")
st.header("Votes")
a=st.number_input("Enter the Votes of the Restaurent",0,10000)
st.header("COST FOR 2 MEMBERS")
b=st.number_input("ENTER THE AVERAGE COST FOR 2 MEMBERS",0,10000)
st.header("PRICE RANGE")
c=st.number_input("ENTER THE RICE RANGE",1,4)
st.header("HAS TABLE BOOKING")
d=st.number_input("ENTER THE AVALABILITY OF TABLE BOOKING",0,1)
st.header("HAS ONLINE DELRVIRY")
e=st.number_input("IS ONLINE DELIEVRY AVILABE ? ",0,1)


s=st.button("PREDICT RATING")
if(s):
    Votes=a
    Average_Cost_for_two=b
    Price_range=c
    Has_Table_booking_0=d
    Has_Online_delivery_0=e
    y_p=data.predict([[Votes,Average_Cost_for_two,Price_range,Has_Table_booking_0,Has_Online_delivery_0]])
    st.write("Predicted Value:",y_p)
    

    




