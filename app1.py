from re import A
import numpy as np
import streamlit as st
import pandas as pd
import cv2
from collections import Counter
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Dense, Dropout, Flatten # type: ignore
from tensorflow.keras.layers import Conv2D # type: ignore
from tensorflow.keras.layers import MaxPooling2D # type: ignore

# Read data
df = pd.read_csv("muse_v3.csv")
df['link'] = df['lastfm_url']
df['name'] = df['track']
df['emotional'] = df['number_of_emotion_tags']
df['pleasant'] = df['valence_tags']
df = df[['name','emotional','pleasant','link','artist']]
df = df.sort_values(by=["emotional","pleasant"])
df.reset_index(inplace=True)
df_sad = df[:18000][['name', 'emotional', 'pleasant', 'link', 'artist']]
df_fear = df[18000:36000][['name', 'emotional', 'pleasant', 'link', 'artist']]
df_angry = df[36000:54000][['name', 'emotional', 'pleasant', 'link', 'artist']]
df_neutral = df[54000:72000][['name', 'emotional', 'pleasant', 'link', 'artist']]
df_happy = df[72000:][['name', 'emotional', 'pleasant', 'link', 'artist']]

# Define preprocessing function
def pre(l):
    result = [item for items, c in Counter(l).most_common() for item in [items] * c]
    ul = []
    for x in result:
        if x not in ul:
            ul.append(x)
    return ul

# Load model
model = Sequential()
# Add layers to the model

# Define emotion dictionary
emotion_dict = {0:"Angry",1:"Disgusted",2:"Fearful",3:"Happy",4:"Neutral",5:"Sad",6:"Surprised"}

# Initialize webcam
cv2.ocl.setUseOpenCL(False)
cap = cv2.VideoCapture(0)

# Streamlit UI
st.markdown("<h2 style='text-align: center; color: white;'><b>TuneAura</b></h2>", unsafe_allow_html=True)
st.markdown("<h5 style='text-align: center; color: grey;'><b>Click on the name of the recommended song to reach website</b></h5>", unsafe_allow_html=True)
col1, col2, col3 = st.columns(3)

# Webcam access HTML code
html_code = """
<div id="video-container">
  <video id="video" width="640" height="480" autoplay></video>
</div>
<script>
navigator.mediaDevices.getUserMedia({ video: true })
  .then(stream => {
    const video = document.getElementById('video');
    video.srcObject = stream;
  })
  .catch(err => {
    console.error('Error accessing webcam:', err);
  });

const video = document.getElementById('video');

// Function to process webcam frames and detect emotions
function detectEmotions() {
  const canvas = document.createElement('canvas');
  const context = canvas.getContext('2d');
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  context.drawImage(video, 0, 0, canvas.width, canvas.height);
  const imageData = context.getImageData(0, 0, canvas.width, canvas.height);

  // TODO: Implement emotion detection logic using TensorFlow.js or OpenCV.js

  // Example: Display placeholder text
  const placeholderText = document.createElement('p');
  placeholderText.textContent = 'Emotion detected: Happy';
  document.body.appendChild(placeholderText);
}

// Periodically call detectEmotions function
setInterval(detectEmotions, 1000); // Adjust interval as needed
</script>
"""
st.markdown(html_code, unsafe_allow_html=True)

# Emotion detection logic
list = []
with col1:
    pass
with col2:
    if st.button("Start Webcam"):
        st.write("Emotion detected: Happy (Placeholder)")
        cap = cv2.VideoCapture(0)
        count = 0
        list = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            face = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
            count = count + 1

            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y - 50), (x + w, y + h + 10), (255, 0, 0), 2)
                roi_gray = gray[y:y + h, x:x + w]
                cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
                prediction = model.predict(cropped_img)
                max_index = int(np.argmax(prediction))
                list.append(emotion_dict[max_index])
                cv2.putText(frame, emotion_dict[max_index],(x+20,y-60),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2,cv2.LINE_AA)
                cv2.imshow('Video',cv2.resize(frame,(1000,700),interpolation=cv2.INTER_CUBIC))
                if cv2.waitKey(1) & 0xFF == ord('x'):
                    break
                if count >= 40:
                    break

        cap.release()
        cv2.destroyAllWindows()
        list = pre(list)

with col3:
    pass

# Function to select recommended songs based on detected emotions
def select_songs(list):
    data = pd.DataFrame()
    if len(list) > 0:
        for emotion in list:
            if emotion == 'Neutral':
                data = pd.concat([data, df_neutral[['name', 'artist', 'link']].sample(n=30)])
            elif emotion == 'Angry':
                data = pd.concat([data, df_angry[['name', 'artist', 'link']].sample(n=30)])
            elif emotion == 'Fear':
                data = pd.concat([data, df_fear[['name', 'artist', 'link']].sample(n=30)])
            elif emotion == 'Happy':
                data = pd.concat([data, df_happy[['name', 'artist', 'link']].sample(n=30)])
            elif emotion == 'Sad':
                data = pd.concat([data, df_sad[['name', 'artist', 'link']].sample(n=30)])
    return data

# Generate recommended songs
recommended_songs = select_songs(list)

# Display recommended songs
st.write("")
st.markdown("<h5 style='text-align: center; color: grey;'><b>Recommended songs with artist names</b></h5>", unsafe_allow_html=True)
st.write("---------------------------------------------------------------------------------------------------------------------")
try:
    for i, (name, artist, link) in recommended_songs.iterrows():
        st.markdown(f"<h4 style='text-align:


