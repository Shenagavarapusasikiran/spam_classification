import random
import requests
import streamlit as st
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer

# Download necessary NLTK data
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')

# Define preprocess function
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess(text):
    tokens = nltk.word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word.lower()) for word in tokens if word.lower() not in stop_words]
    return ' '.join(tokens)

# Load model and vectorizer
with open('spam_classifier11.pkl', 'rb') as model_file:
    vectorizer, model = pickle.load(model_file)

# Function to classify email
def classify_email(message):
    processed_message = preprocess(message)
    message_vector = vectorizer.transform([processed_message])
    prediction = model.predict(message_vector)
    return prediction[0]

# Streamlit interface
def main():
    st.title('Spam Classifier')

    # Input for email message
    message = st.text_area('Enter email message:', height=200)

    # Input for email URL (optional)
    url = st.text_input('Enter URL for email content (optional):')

    if st.button('Classify'):
        if message.strip() == '' and url.strip() == '':
            st.warning('Please enter an email message or provide a URL.')
        elif url.strip() != '':
            # Fetch email content from URL
            try:
                response = requests.get(url)
                if response.status_code == 200:
                    email_content = response.text
                    prediction = classify_email(email_content)
                    st.write(f'Classification result: {prediction}')
                else:
                    st.error(f'Error fetching email content: Status code {response.status_code}')
            except requests.exceptions.RequestException as e:
                st.error(f'Error fetching email content: {str(e)}')
        else:
            # Classify based on entered message
            prediction = classify_email(message)
            st.write(f'Classification result: {prediction}')

if __name__ == '__main__':
    main()
