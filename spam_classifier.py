import streamlit as st
import pickle

model = pickle.load(open("spam.pkl", "rb"))
cv = pickle.load(open("vec.pkl", "rb"))

st.title("Email Spam Classification App")
st.write("This is a Machine Learning app to classify emails as Spam or Not Spam.")
st.subheader("Spam Classification")

user_input = st.text_area("Enter the email body for Spam check.", height=200)

if st.button("Check"):
    if user_input:
        data = [user_input]
        vec = cv.transform(data).toarray()
        result = model.predict(vec)
        print(result)
        if result[0] == 0:
            st.success("This is not a Spam mail.")
        else:
            st.error("This is a Spam mail.")
    else:
        st.write("Please enter an email body for Spam check.")