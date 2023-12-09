import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

# Load the data
data = pd.read_csv('mail_data.csv')

# Preprocess data
data['Category'] = data['Category'].map({'spam': 0, 'ham': 1})

# Split the data
x = data['Message']
y = data['Category']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=42)

# Train the model
reg = LogisticRegression()
feature_extraction = TfidfVectorizer(min_df=1, stop_words='english')
x_train_features = feature_extraction.fit_transform(x_train)
x_test_features = feature_extraction.transform(x_test)
y_train = y_train.astype('int')
y_test = y_test.astype('int')
reg.fit(x_train_features, y_train)

# Streamlit app
st.title("Spam/Ham Classification App")

# Text input for new mail
input_mail = st.text_area("Enter the mail text:", "Lucrative stipend of INR 1,00,000 & potential PPI/PPS")

# Button to predict
if st.button("Predict"):
    input_mail_features = feature_extraction.transform([input_mail])
    prediction = reg.predict(input_mail_features)

    # Display prediction
    st.success("Prediction:")
    if prediction[0] == 1:
        st.write("HAM")
    else:
        st.write("SPAM")
