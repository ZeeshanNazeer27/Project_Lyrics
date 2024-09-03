import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from transformers import AutoTokenizer, pipeline
from music21 import note, stream, midi, instrument as m21_instrument
from io import BytesIO
import os
import tempfile

# Load pre-trained models
lyrics_model = load_model('lyrics_emotion_rnn_model.h5')
music_model = load_model('music_model.h5')

# Load tokenizer and emotion classifier
tokenizer = AutoTokenizer.from_pretrained("SamLowe/roberta-base-go_emotions")
classifier = pipeline("text-classification", model="SamLowe/roberta-base-go_emotions", tokenizer=tokenizer)

# Emotion to lyrics dictionary (English and Urdu)
emotion_lyrics = {
    "joy": {
        "english": "Sunshine in my heart, a world so bright\nLaughter fills the air, everything feels right\nDancing through the day, with a smile so wide\nHappiness abounds, joy I can't hide",
        "urdu": "دل دل پاکستان\nجان جان پاکستان\nکشمیر پاکستان\nسرحد پاکستان\nپنجاب پاکستان\nسندھ پاکستان\nبلوچستان پاکستان"
    },
    "sadness": {
        "english": "Tears fall like rain, on this gloomy day\nMemories linger, as clouds turn gray\nHeartache whispers, in the silent night\nLoneliness embraces, as hope fades from sight",
        "urdu": "ہم دیکھیں گے\nلازم ہے کہ ہم بھی دیکھیں گے\nوہ دن کہ جس کا وعدہ ہے\nجو لوح ازل پہ لکھا ہے\nجب ظلم و ستم کے کوہ گراں\nروئی کی طرح اڑ جائیں گے"
    },
    "anger": {
        "english": "Fire in my veins, rage burns so strong\nFrustration builds, everything feels wrong\nShattered patience, words sharp as knives\nAnger consumes, as tension arrives",
        "urdu": "میں بغاوت ہوں، میں بغاوت ہوں\nمیرے لب پہ آزادی کا ترانہ ہے\nمیں وہ آواز ہوں، جو دبائی نہیں جا سکتی\nمیں وہ آگ ہوں، جو بجھائی نہیں جا سکتی"
    },
    "fear": {
        "english": "Shadows lurk, in every corner\nHeart races fast, like a frightened mourner\nDarkness closing in, can't catch my breath\nFear grips tight, like the touch of death",
        "urdu": "ڈر لگتا ہے\nجب آنکھ لگتی ہے\nسو جاتے ہیں\nجاگتے ہیں تو\nڈر لگتا ہے"
    },
    "love": {
        "english": "Your smile lights up, my whole universe\nHearts intertwined, for better or worse\nLove flows freely, like a gentle stream\nIn your arms I'm home, living our dream",
        "urdu": "اجے میرے پیار کے صفحے پر\nلکھ دے تیرے نام کا اک حرف\nجو مٹ نہ سکے\nجو کٹ نہ سکے\nایسا کوئی حرف لکھ دے"
    },
    "surprise": {
        "english": "Unexpected turns, life's grand surprise\nWonder fills my heart, opens up my eyes\nShock and awe combine, in this moment rare\nAmazement takes hold, beyond compare",
        "urdu": "وہ کیا گنگ گھٹا تھی\nکہ برس نہ سکی\nہم پیاسے رہے\nاور پیاسے رہے\nوہ کس طرح کی آرزو تھی\nکہ مر نہ سکی"
    }
}

# Helper function to encode features
def encode_features(instrument, tempo, time_signature, key):
    instrument_dict = {'Piano': 0, 'Guitar': 1, 'Violin': 2, 'Cello': 3, 'Drums': 4, 'Harp': 5}
    time_signature_dict = {'2/4': 0, '3/4': 1, '4/4': 2, '5/4': 3, '6/4': 4, '7/4': 5}
    key_dict = {'C': 0, 'G': 1, 'Am': 2, 'Em': 3, 'D': 4, 'F': 5}

    encoded_features = [
        instrument_dict.get(instrument, -1),
        tempo,
        time_signature_dict.get(time_signature, -1),
        key_dict.get(key, -1)
    ]
    
    return np.array(encoded_features).reshape(1, -1)

# Helper function to generate melody
def generate_melody(features):
    X_new = encode_features(*features)
    predicted_melody = music_model.predict(X_new)
    
    melody_stream = stream.Stream()

    # Set the instrument based on user selection
    instrument_mapping = {
        'Piano': m21_instrument.Piano(),
        'Guitar': m21_instrument.Guitar(),
        'Violin': m21_instrument.Violin(),
        'Cello': m21_instrument.Violoncello(),
        'Drums': m21_instrument.Percussion(),
        'Harp': m21_instrument.Harp()
    }

    # Add the selected instrument to the stream
    selected_instrument = features[0]
    melody_stream.append(instrument_mapping[selected_instrument])

    for pitch in predicted_melody[0]:
        melody_stream.append(note.Note(int(pitch)))
    
    return melody_stream

# Helper function to save MIDI file
def save_midi(melody_stream):
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mid') as temp_file:
        melody_stream.write('midi', temp_file.name)
        with open(temp_file.name, 'rb') as file:
            midi_content = file.read()
    os.remove(temp_file.name)
    return midi_content

# Streamlit app interface
st.title('Lyrics and Melody Generator')

# Step 1: Language selection
language = st.selectbox('Select Language', ['English', 'Urdu'])

# Step 2: Emotion selection
emotion = st.selectbox('Select Emotion', ['joy', 'sadness', 'anger', 'fear', 'love', 'surprise'])

# Step 3: Generate and display lyrics
if st.button('Generate Lyrics'):
    lyrics = emotion_lyrics[emotion][language.lower()]
    st.subheader('Generated Lyrics')
    st.write(lyrics)

# Step 4: Melody generation options
st.subheader('Melody Options')
instrument = st.selectbox('Select Instrument', ['Piano', 'Guitar', 'Violin', 'Cello', 'Drums', 'Harp'])
tempo = st.slider('Select Tempo (BPM)', min_value=60, max_value=180, value=120)
time_signature = st.selectbox('Select Time Signature', ['2/4', '3/4', '4/4', '5/4', '6/4', '7/4'])
key = st.selectbox('Select Key', ['C', 'G', 'Am', 'Em', 'D', 'F'])

# Step 5: Generate and download melody
if st.button('Generate Melody'):
    features = [instrument, tempo, time_signature, key]
    melody_stream = generate_melody(features)
    
    midi_file = save_midi(melody_stream)
    st.download_button('Download MIDI', midi_file, file_name='generated_melody.mid', mime='audio/midi')
