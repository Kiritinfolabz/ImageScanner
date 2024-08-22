import streamlit as st
import pandas as pd

st.set_page_config(page_icon=":exclamation:",
                   page_title="AI-Driven Visual Insights for Wellness and Sustainable Harvests")

# Handling page redirection based on URL parameters
query_params = st.experimental_get_query_params()
page = query_params.get("page", ["login"])[0]

if page == "login":
    st.title("LOGIN")

    # To hide sidebar
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
    password = st.text_input("Enter Password", type="password")

    # Check if the user clicked the login button
    try:
        if st.button("Login"):
            if data[data["Email"] == email]["Password"].values[0] == password:
                st.success("Login successful")
                # Redirect to the main page or any other page
                st.experimental_set_query_params(page="main")
                st.experimental_rerun()
            else:
                st.error("Invalid Email Or Password")
    except:
        st.warning("Enter email And Password")

    # Link to redirect to the signup page
    link = '[Register](?page=signup)'
    st.markdown(link, unsafe_allow_html=True)

elif page == "signup":
    st.title("SIGNUP")
    st.write("Welcome to the Signup Page!")

    # Add form fields for user registration
    new_email = st.text_input("Enter your email")
    new_password = st.text_input("Enter your password", type="password")

    if st.button("Register"):
        # Add your registration logic here
        st.success("Registration successful! Please log in.")

    # Link to go back to the login page
    link = '[Go back to Login](?page=login)'
    st.markdown(link, unsafe_allow_html=True)

elif page == "main":
    st.title("Main Page")
    st.write("Welcome to the main page after login!")

    # Example: Link to redirect to another page in the 'pages' directory
    other_page_link = '[Go to Other Page](?page=other_page)'
    st.markdown(other_page_link, unsafe_allow_html=True)

elif page == "other_page":
    st.title("Other Page")
    st.write("This is another page in your 'pages' directory.")
