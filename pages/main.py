import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from PIL import Image
from keras.models import load_model
from keras.preprocessing import image
from streamlit_option_menu import option_menu

st.set_page_config(page_icon=":exclamation:",page_title="AI-Driven Visual Insights for Wellness and Sustainable Harvests")
st.set_option('deprecation.showPyplotGlobalUse', False)
st.markdown(
    """
        <style>
    [data-testid="stSidebarNavLink"]{
        visibility: hidden;
    }
         .reportview-container {
            margin-top: -2em;
        }

        .stDeployButton {display:none;}
        footer {visibility: hidden;}
        #stDecoration {display:none;}
        footer {visibility: hidden;}
    [data-testid="collapsedControl"] {
        display: none
    }
    </style>
    """,
    unsafe_allow_html=True
)
with st.sidebar:
    choose = option_menu("AI", ["About", "Data Analysis (Fracture)","Data Visualization (Fracture)","Bone Fracture (AI + ML)","Data Analysis (Covid)","Data Visualization (Covid)", "Covid19 (AI + ML)","Data Analysis (Tumor)","Data Visualization (Tumor)","Brain Tumor (AI + ML)","Tomato_leaf_Disease (AI + ML)","Crop Data anaysis and visualziation","Feedback","LOGOUT"],
                         icons=['house','recycle', 'clipboard2-data-fill','activity ', 'virus2 ','align-middle','virus2', 'braces','bar-chart-fill','hourglass-split','shield-slash-fill','tree-fill','person-lines-fill','arrow-right-square-fill'],
                         menu_icon="app-indicator", default_index=0,
                         styles={
        "container": {"padding": "5!important", "background-color": "#fafafa"},
        "icon": {"color": "black", "font-size": "25px"},
        "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px", "--hover-color": "#eee"},
        "nav-link-selected": {"background-color": "#42f2b5"},
    }
    )

def about():
    st.title("About AI-Driven Visual Insights for Wellness and Sustainable Harvests")

    st.write(
        "Welcome to our AI-Driven Visual Insights platform, where cutting-edge technology meets the pressing needs of both personal well-being and environmental sustainability. Our platform is designed to empower individuals and organizations with actionable insights derived from advanced AI algorithms, revolutionizing the way we approach wellness and agricultural practices.")

    st.header("Wellness Insights:")
    st.write(
        "In the realm of wellness, our platform offers personalized analyses of health metrics to support individuals in their journey towards healthier living. By leveraging AI-driven visualizations, we provide real-time feedback on various aspects of well-being, including stress levels, sleep patterns, dietary habits, and physical activity. Our goal is to equip users with the knowledge and tools they need to make informed decisions about their health and lifestyle choices, ultimately leading to improved overall wellness and quality of life.")

    st.header("Sustainable Harvests Insights:")
    st.write(
        "In the agricultural sector, our platform plays a crucial role in promoting sustainable practices and maximizing crop yields. Through advanced AI algorithms and image processing techniques, we offer insights into crop health, early detection of diseases or stress, and optimal resource allocation. By enabling farmers and land managers to make data-driven decisions, we aim to enhance agricultural productivity while minimizing environmental impact, ensuring a more sustainable future for generations to come.")


def fracture():


    # Load the model
    model = load_model('dataset/xray_fracture_detection_model.h5')  # Load the model you saved

    # Define the classes
    classes = ['Fracture', 'Normal']

    def predict(image_path):
        print("Loading and preprocessing image...")
        img = image.load_img(image_path, target_size=(150, 150))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0  # Normalize the image
        print("Image loaded and preprocessed. Shape:", img_array.shape)

        print("Making prediction...")
        prediction = model.predict(img_array)
        print("Prediction received. Shape:", prediction.shape)

        predicted_class = classes[int(np.round(prediction)[0][0])]
        return predicted_class

    # Streamlit app
    def main():
        st.title('X-ray Image Classification')
        st.write('Upload an X-ray image to classify it as "Fracture" or "Normal".')

        uploaded_file = st.file_uploader("Choose an X-ray image...", type=['jpg', 'jpeg', 'png'])

        if uploaded_file is not None:
            # Display the uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption='Uploaded X-ray Image.', use_column_width=True)

            # Make a prediction
            predicted_class = predict(uploaded_file)
            st.write('Prediction:', predicted_class)

    if __name__ == "__main__":
        main()




def covid():
    model = load_model('dataset/covid_classification_model.h5')

    # Define image dimensions
    img_width, img_height = 150, 150

    # Define class labels
    labels = ['Covid', 'Normal', 'Viral Pneumonia']

    # Function to preprocess uploaded image
    def preprocess_image(image):
        img = image.resize((img_width, img_height))
        img_array = np.array(img)
        img_array = np.expand_dims(img_array, axis=0)
        return img_array / 255.0

    # Function to make predictions
    def predict(image):
        img_array = preprocess_image(image)
        predictions = model.predict(img_array)
        predicted_label_index = np.argmax(predictions)
        return labels[predicted_label_index]

    # Streamlit app
    def main():
        st.title('Chest X-Ray Image Classification')
        st.write('Upload a chest X-ray image and get the prediction')

        # File uploader
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

        if uploaded_file is not None:
            # Display uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption='Uploaded Image', use_column_width=True)

            # Make prediction when button is clicked
            if st.button('Predict'):
                prediction = predict(image)
                st.write('Prediction:', prediction)

    if __name__ == '__main__':
        main()

def brain():


    # Load the saved model
    model = load_model('dataset/latest_brain_tumor.h5')

    # Preprocess the input image
    # Define the classes
    classes = ['Tumor', 'Normal']

    def predict(image_path):
        print("Loading and preprocessing image...")
        img = image.load_img(image_path, target_size=(150, 150))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0  # Normalize the image
        print("Image loaded and preprocessed. Shape:", img_array.shape)

        print("Making prediction...")
        prediction = model.predict(img_array)
        print("Prediction received. Shape:", prediction.shape)

        predicted_class = classes[int(np.round(prediction)[0][0])]
        return predicted_class

    # Streamlit app
    def main():
        st.title('X-ray Image Classification')
        st.write('Upload an X-ray image to classify it as "Tumor" or "Normal".')

        uploaded_file = st.file_uploader("Choose an X-ray image...", type=['jpg', 'jpeg', 'png'])

        if uploaded_file is not None:
            # Display the uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption='Uploaded X-ray Image.', use_column_width=True)

            # Make a prediction
            predicted_class = predict(uploaded_file)
            st.write('Prediction:', predicted_class)

    if __name__ == "__main__":
        main()

def tomato():
    import streamlit as st
    from PIL import Image
    from tensorflow.keras.models import load_model
    import numpy as np

    # Load the saved model
    model_path = 'dataset/model_inception.h5'
    model = load_model(model_path)

    # Define class labels
    class_labels = [
        'Tomato___healthy',
        'Tomato___Tomato_mosaic_virus',
        'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
        'Tomato___Spider_mites Two-spotted_spider_mite',
        'Tomato___Target_Spot',
        'Tomato___Septoria_leaf_spot',
        'Tomato___Late_blight',
        'Tomato___Leaf_Mold',
        'Tomato___Bacterial_spot',
        'Tomato___Early_blight'
    ]

    # Function to preprocess the image
    def preprocess_image(img):
        img = img.resize((224, 224))  # Resize image to 224x224 pixels
        img = img.convert('RGB')  # Convert image to RGB format
        img = np.array(img)  # Convert image to numpy array
        img = img / 255.0  # Normalize pixel values
        img = np.expand_dims(img, axis=0)  # Add batch dimension
        return img



    # Function to predict the image
    def predict_image(img):
        # Preprocess the image
        img = preprocess_image(img)

        # Make prediction
        prediction = model.predict(img)
        predicted_class_index = np.argmax(prediction, axis=1)[0]
        predicted_class_label = class_labels[predicted_class_index]
        confidence = prediction[0][predicted_class_index]

        return predicted_class_label, confidence

    # Streamlit app
    st.title('Image Classification with InceptionV3')

    # Upload image
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        # Make prediction
        predicted_class, confidence = predict_image(image)
        st.write('Prediction:', predicted_class)
        st.write('Confidence:', confidence)
def contact():
    st.title("INFO")
    st.write("Please fill out the form below to get in touch with us.")

    # Input fields
    name = st.text_input("Your Name")
    email = st.text_input("Your Email")
    message = st.text_area("Message")

    # Submit button
    if st.button("Submit"):
        if name and email and message:
            st.success("Thank you! Your message has been submitted.")
            # You can add code here to handle the submission, such as sending an email or saving to a database
        else:
            st.error("Please fill out all fields.")
def covid_visual():
    df = pd.read_csv("dataset/CVD_cleaned.csv")

    # Title of the web app
    st.title('Health Data Visualization')

    # Title for plot selection
    st.subheader('Select Plot Type')

    # Dropdown for plot selection
    plot_type = st.selectbox('Plot Type', ['Bar Chart', 'Histogram', 'Pie Chart', 'Scatter Plot', 'Box Plot'])

    # Plot based on the selected type
    if plot_type == 'Bar Chart':
        selected_column = st.selectbox('Select Column for Bar Chart', df.columns)
        plt.figure(figsize=(10, 6))
        sns.countplot(data=df, x=selected_column)
        plt.title(f'{selected_column} Distribution')
        plt.xlabel(selected_column)
        plt.ylabel('Count')
        st.pyplot()

    elif plot_type == 'Histogram':
        selected_column = st.selectbox('Select Column for Histogram', df.columns)
        plt.figure(figsize=(10, 6))
        plt.hist(df[selected_column], bins=5, color='skyblue', edgecolor='black')
        plt.title(f'{selected_column} Distribution')
        plt.xlabel(selected_column)
        plt.ylabel('Frequency')
        st.pyplot()

    elif plot_type == 'Pie Chart':
        selected_column = st.selectbox('Select Column for Pie Chart', df.columns)
        plt.figure(figsize=(8, 8))
        df[selected_column].value_counts().plot(kind='pie', autopct='%1.1f%%', colors=['lightblue', 'lightcoral'])
        plt.title(f'{selected_column} Distribution')
        plt.ylabel('')
        st.pyplot()

    elif plot_type == 'Scatter Plot':
        x_column = st.selectbox('Select X Column for Scatter Plot', df.columns)
        y_column = st.selectbox('Select Y Column for Scatter Plot', df.columns)
        hue_column = st.selectbox('Select Hue Column for Scatter Plot', df.columns)
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=df, x=x_column, y=y_column, hue=hue_column, palette='coolwarm')
        plt.title(f'{y_column} vs {x_column} by {hue_column}')
        plt.xlabel(x_column)
        plt.ylabel(y_column)
        st.pyplot()

    elif plot_type == 'Box Plot':
        selected_columns = st.multiselect('Select Columns for Box Plot', df.columns)
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=df[selected_columns], palette='Set2')
        plt.title('Box Plot')
        plt.ylabel('Values')
        st.pyplot()




def covid_analysis():
    df = pd.read_csv("dataset/CVD_cleaned.csv")

    # Title of the web app
    st.subheader("COVID DATASET")
    st.dataframe(df)

    st.subheader("Null Values by Columns")
    st.write(df.isnull().sum())

    st.subheader("Shape of Dataset")
    st.write("ROWS AND COLUMNS")
    st.subheader(df.shape)

    st.subheader("NAME OF COLUMNS")
    st.write(df.columns)


def data_analysis_fracture():
    st.subheader("BONE FRACTURE DATASET")
    df = pd.read_csv("C:/Users/LENOVO/Downloads/dataset.csv")
    st.dataframe(df)

    st.subheader("SHAPE OF DATASET")

    st.write("ROWS AND COLUMNS")
    st.subheader(df.shape)

    st.subheader("COLUMN NAMES")
    st.write(df.columns)
    st.subheader("CHECK NULL VALUES")
    st.write(df.isnull().sum())

def data_visualization_fracture():

    # Load the dataset
    df = pd.read_csv("C:/Users/LENOVO/Downloads/dataset.csv")

    # Title of the web app
    st.title('FRACTURE Data Visualization')
    st.title('Fracture Count by Body Part')

    # Scatter plot for each body part against fracture count
    plt.figure(figsize=(10, 6))

    # Scatter plot for hand
    plt.scatter(df['hand'], df['fracture_count'], label='Hand', alpha=0.5)
    # Scatter plot for leg
    plt.scatter(df['leg'], df['fracture_count'], label='Leg', alpha=0.5)
    # Scatter plot for hip
    plt.scatter(df['hip'], df['fracture_count'], label='Hip', alpha=0.5)
    # Scatter plot for shoulder
    plt.scatter(df['shoulder'], df['fracture_count'], label='Shoulder', alpha=0.5)

    plt.xlabel('Body Part Measurement')
    plt.ylabel('Fracture Count')
    plt.title('Fracture Count by Body Part')
    plt.legend()
    st.pyplot()

    # Histogram of Fracture Counts
    st.subheader('Histogram of Fracture Counts')
    plt.figure(figsize=(8, 6))
    sns.histplot(df['fracture_count'], bins=20, kde=True)
    plt.xlabel('Fracture Count')
    plt.ylabel('Frequency')
    st.pyplot()

    # Scatter Plot Matrix
    st.subheader('Scatter Plot Matrix')
    sns.pairplot(df[['fracture_count', 'hand', 'leg', 'hip', 'shoulder']], diag_kind='kde')
    st.pyplot()

    # Box Plot of Fracture Counts by Categorical Variables
    st.subheader('Box Plot of Fracture Counts by Categorical Variables')
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x='fractured', y='fracture_count')
    plt.xlabel('Fractured')
    plt.ylabel('Fracture Count')
    st.pyplot()

def data_visualization_brain():
    # Load the dataset
    df = pd.read_csv("dataset/brain.csv")

    # Title of the web app
    st.title('Data Analysis')

    # Bar Chart for Diagnosis
    st.subheader('Diagnosis Distribution')
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, x='Diagnosis')
    plt.xlabel('Diagnosis')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    st.pyplot()

    # Histogram for Age
    st.subheader('Age Distribution')
    plt.figure(figsize=(10, 6))
    sns.histplot(df['Age'], bins=20, kde=True)
    plt.xlabel('Age')
    plt.ylabel('Frequency')
    st.pyplot()

    # Pie Chart for Gender
    st.subheader('Gender Distribution')
    plt.figure(figsize=(8, 8))
    df['Gender'].value_counts().plot(kind='pie', autopct='%1.1f%%', colors=['lightblue', 'lightcoral'])
    plt.title('Gender Distribution')
    plt.ylabel('')
    st.pyplot()

def data_analysis_brain():
    df = pd.read_csv("dataset/brain.csv")
    st.subheader("Brain Tumor Dataset")
    st.dataframe(df)

    st.subheader("Column Names")
    st.write(df.columns)

    st.subheader("Check Null values")
    st.write(df.isnull().sum())

    # Summary Statistics
    st.subheader('Summary Statistics for Age')
    st.write(df['Age'].describe())

    st.subheader('Counts of Different Categories')

    # Counts of Gender
    st.write('Gender Distribution:')
    st.write(df['Gender'].value_counts())

    # Counts of Diagnosis
    st.write('Diagnosis Distribution:')
    st.write(df['Diagnosis'].value_counts())

    # Counts of SourceCode
    st.write('SourceCode Distribution:')
    st.write(df['SourceCode'].value_counts())

def crop():

    # Load the dataset
    df = pd.read_csv("dataset/final_dataSet_nonan.csv")

    # Title of the web app
    st.title('Crop Data Analysis')

    # Summary Statistics
    st.subheader('Summary Statistics')
    st.write(df.describe())

    # Distribution Analysis
    st.subheader('Distribution Analysis')

    # Histogram for Area
    st.write('Histogram for Area')
    plt.hist(df['Area'], bins=20)
    st.pyplot()

    # Histogram for Rain
    st.write('Histogram for Rainfall')
    plt.hist(df['Rain'], bins=20)
    st.pyplot()

    # Histogram for Production
    st.write('Histogram for Production')
    plt.hist(df['Production'], bins=20)
    st.pyplot()

    # Relationship Analysis
    st.subheader('Relationship Analysis: Rainfall vs Production')
    plt.scatter(df['Rain'], df['Production'])
    plt.xlabel('Rainfall')
    plt.ylabel('Production')
    st.pyplot()


if choose == "About":
    about()
elif choose == "Data Analysis (Fracture)":
    data_analysis_fracture()
elif choose =="Data Visualization (Fracture)":
    data_visualization_fracture()
elif choose == "Bone Fracture (AI + ML)":
    fracture()
elif choose =="Data Analysis (Covid)":
    covid_analysis()
elif choose =="Data Visualization (Covid)":
    covid_visual()
elif choose == "Covid19 (AI + ML)":
    covid()
elif choose =="Brain Tumor (AI + ML)":
    brain()
elif choose == "Tomato_leaf_Disease (AI + ML)":
    tomato()
elif choose == "Feedback":
    contact()
elif choose =="Data Analysis (Tumor)":
    data_analysis_brain()
elif choose == "Data Visualization (Tumor)":
    data_visualization_brain()
elif choose == "LOGOUT":
    st.markdown(f'<meta http-equiv="refresh" content="2;url=https://imagescanner.streamlit.app/Login">', unsafe_allow_html=True)
    st.header("Redirecting...")
elif choose =="Crop Data anaysis and visualziation":
    crop()
