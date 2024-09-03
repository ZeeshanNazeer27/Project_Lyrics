import pandas as pd
from nltk.tokenize import word_tokenize
import re
from nltk.corpus import stopwords
from transformers import pipeline, AutoTokenizer
from collections import Counter
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import nltk

# Load and preprocess data
data = pd.read_csv('songs.csv')
data.head()

nltk.download('punkt')
nltk.download('stopwords')

stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = word_tokenize(text)
    cleaned_tokens = [token.lower() for token in tokens if token.lower() not in stop_words]
    return " ".join(cleaned_tokens)

data['cleaned_lyrics'] = data['Lyrics'].apply(clean_text)

# Load tokenizer and classifier
model_name = "SamLowe/roberta-base-go_emotions"
tokenizer = AutoTokenizer.from_pretrained(model_name)
classifier = pipeline("text-classification", model=model_name, tokenizer=tokenizer)

def chunk_text_with_tokenizer(text, tokenizer, max_length=512):
    """Split text into chunks based on the tokenizer's tokenization."""
    tokens = tokenizer(text, truncation=True, add_special_tokens=False)['input_ids']
    chunks = [tokens[i:i + max_length] for i in range(0, len(tokens), max_length)]
    return [tokenizer.decode(chunk, skip_special_tokens=True) for chunk in chunks]

def label_emotion(lyrics, tokenizer, max_length=512):
    chunks = chunk_text_with_tokenizer(lyrics, tokenizer, max_length)
    results = []
    for chunk in chunks:
        try:
            # Encode and prepare inputs for the model
            inputs = tokenizer(chunk, return_tensors='pt', truncation=True, padding='max_length', max_length=max_length)
            result = classifier(inputs['input_ids'])[0]['label']
            results.append(result)
        except IndexError:
            print(f"Skipping chunk due to IndexError: {chunk}")
            pass
    if not results:
        return 'unknown'

    emotion_count = Counter(results)
    most_common_emotion = emotion_count.most_common(1)[0][0]
    return most_common_emotion

data['emotion'] = data['cleaned_lyrics'].apply(lambda lyrics: label_emotion(lyrics, tokenizer))

print(data)

# Prepare data for RNN model
tokenizer = Tokenizer(num_words=10000)
tokenizer.fit_on_texts(data['cleaned_lyrics'])
sequences = tokenizer.texts_to_sequences(data['cleaned_lyrics'])

max_length = 100
padded_sequences = pad_sequences(sequences, maxlen=max_length, padding='post')

# Encode the emotion labels
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(data['emotion'])

X_train, X_test, y_train, y_test = train_test_split(padded_sequences, encoded_labels, test_size=0.2, random_state=42)

# Define and train the RNN model
model = Sequential([
    Embedding(input_dim=10000, output_dim=128, input_length=max_length),
    LSTM(256, return_sequences=True),
    Dropout(0.2),
    LSTM(128),
    Dropout(0.2),
    Dense(256, activation='relu'),
    Dropout(0.2),
    Dense(128, activation='relu'),
    Dense(len(label_encoder.classes_), activation='softmax')  # For multi-class classification
])

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

history = model.fit(X_train, y_train, epochs=15, validation_data=(X_test, y_test), batch_size=32)

# Save the model if needed
model.save('lyrics_emotion_rnn_model.h5')
