import streamlit as st
import pickle
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer


loaded_model = pickle.load(open('hate_model.pkl', 'rb'))
vectorizer = pickle.load(open('tfidf_vectorizer.pkl', 'rb'))

def preprocess_text(text):
   
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word not in stop_words]
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(word) for word in filtered_tokens]
    preprocessed_text = ' '.join(lemmatized_tokens)
    return preprocessed_text

# Function to predict hate speech
def predict_hate_speech(text):
    preprocessed_text = preprocess_text(text)
    vectorized_text = vectorizer.transform([preprocessed_text])
    prediction = loaded_model.predict(vectorized_text)[0]
    return prediction


def main():
    st.title("Hate Speech Detection")

    user_input = st.text_area("Enter text here:")

    if st.button("Predict"):
        if user_input.strip():
            prediction = predict_hate_speech(user_input)
            if prediction == 1:
                st.write("Prediction: Hate Speech")
            else:
                st.write("Prediction: Normal Language")
        else:
            st.warning("Please enter some text.")

if __name__ == "__main__":
    main()
