from flask import Flask, render_template, request
import pickle
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pandas as pd  # Import pandas for DataFrame operations

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

app = Flask(__name__)

# Load model, tokenizer, pad_sequences, and label_encoder
model = pickle.load(open('model.pkl', 'rb'))
tokenizer = pickle.load(open('tokenizer.pkl', 'rb'))
label_encoder = pickle.load(open('label_encoder.pkl', 'rb'))
pad_sequences = pickle.load(open('pad_sequences.pkl', 'rb'))

# Load DataFrame for filtering
merged_df = pd.read_csv('merged_df.csv')  # Replace with the actual file path
print(merged_df.columns)
# Define max_length according to your training setup
MAX_LENGTH = 100

def preprocess_text(text):
    text = text.replace(',', ' ')
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'_', ' ', text)
    word_tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    filtered_text = [word for word in word_tokens if word not in stop_words]
    lemmatizer = WordNetLemmatizer()
    lemmatized_text = [lemmatizer.lemmatize(word) for word in filtered_text]
    return ' '.join(lemmatized_text)

def preprocess_new_text(text, tokenizer, max_length):
    processed_text = preprocess_text(text)
    sequence = tokenizer.texts_to_sequences([processed_text])
    padded_sequence = pad_sequences(sequence, maxlen=max_length, padding='post')
    return padded_sequence

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    disease_info = None
    first_row_str = None

    if request.method == 'POST':
        if 'symptoms' in request.form:
            symptoms = request.form['symptoms']
            
            # Preprocess the input
            preprocessed_symptoms = preprocess_new_text(symptoms, tokenizer, max_length=MAX_LENGTH)
            
            # Make a prediction
            predictions = model.predict(preprocessed_symptoms)
            predicted_class = np.argmax(predictions, axis=1)[0]
            predicted_label = label_encoder.inverse_transform([predicted_class])[0]
            prediction = predicted_label
            
            # Get disease information
            rows = merged_df[merged_df['Disease'].str.contains(prediction,regex=False)]
            if not rows.empty:
    # Get the first row corresponding to the predicted disease
                    first_row = rows.iloc[0]
    
    # Define precautions list
                    precautions = [
                        first_row['Precaution_1'],
                        first_row['Precaution_2'],
                        first_row['Precaution_3'],
                        first_row['Precaution_4']
                    ]
                    
                    # If 'consult doctor' is in precautions, keep only that
                    if 'consult doctor' in precautions:
                        precautions = ['consult doctor']
                    
                    # Create disease_info dictionary
                    disease_info = {
                        "Disease": first_row['Disease'],
                        "Description": first_row['Description'],
                        "Diet": first_row['Diet'],
                        "Medication": first_row['Medication'],
                        "Precautions": precautions
                    }
            else:
                first_row_str = f"No records found with the disease '{prediction}'."
    
    return render_template('index.html', prediction=prediction, disease_info=disease_info, first_row=first_row_str)

if __name__ == '__main__':
    app.run(debug=True)
