import streamlit as st
import pandas as pd
st.set_page_config(page_icon=":exclamation:",page_title="About AI-Driven Visual Insights for Wellness and Sustainable Harvests")

st.title("Welcome:- About AI-Driven Visual Insights for Wellness and Sustainable Harvests")
st.title("LOGIN ")

# To hide side bar we use :-

st.markdown(
    """
        <style>
    [data-testid="stSidebar"]{   
        visibility: hidden;
    }
    [data-testid="stSidebarNavLink"]{
        visibility: hidden;
    }
    </style>
    """,
    unsafe_allow_html=True
)

df = pd.read_csv("data.csv")
data = pd.DataFrame(df)

# Get the user input
email = st.text_input("Enter email")
password = st.text_input("Enter Password",type="password")

# Check if the user clicked the login button
try:
    if st.button("Login"):
        if data[data["Email"] == email]["Password"].values[0] == password:
            st.success("Login successful")
            st.markdown(f'<meta http-equiv="refresh" content="2;url=http://localhost:8501/main">', unsafe_allow_html=True)
            st.header("Redirecting...")
        else:
            st.error("Invalid Email Or Password")
except:
    st.warning("Enter email And Password")


link='[Register](http://localhost:8501/Signup)'
st.markdown(link,unsafe_allow_html=True)
