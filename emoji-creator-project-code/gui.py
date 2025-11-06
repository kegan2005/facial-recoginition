import tkinter as tk
from tkinter import *
import cv2
from PIL import Image, ImageTk
import numpy as np
from pathlib import Path
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D

# Build model structure 
emotion_model = Sequential()
emotion_model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(48,48,1)))
emotion_model.add(Conv2D(64, (3, 3), activation='relu'))
emotion_model.add(MaxPooling2D((2, 2)))
emotion_model.add(Dropout(0.25))

emotion_model.add(Conv2D(128, (3, 3), activation='relu'))
emotion_model.add(MaxPooling2D((2, 2)))
emotion_model.add(Conv2D(128, (3, 3), activation='relu'))
emotion_model.add(MaxPooling2D((2, 2)))
emotion_model.add(Dropout(0.25))

emotion_model.add(Flatten())
emotion_model.add(Dense(1024, activation='relu'))
emotion_model.add(Dropout(0.5))
emotion_model.add(Dense(7, activation='softmax'))

# Resolve model path relative to this script;
model_path = (Path(__file__).resolve().parent / 'model.h5')
if not model_path.exists():
    model_path = Path(__file__).resolve().parent.parent / 'model.h5'
emotion_model.load_weights(str(model_path))

cv2.ocl.setUseOpenCL(False)

emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

script_dir = Path(__file__).resolve().parent
# Check both current directory and parent for the emojis folder
possible_emojis_dirs = [
    script_dir / "emojis" / "emojis",
    script_dir.parent / "emojis" / "emojis",
]
emojis_dir = next((p for p in possible_emojis_dirs if p.exists()), possible_emojis_dirs[0])

# Map emotion index to filename; use available files in the repo.
# For classes without matching files, we will fall back to 'neutral.png'.
emoji_filenames = {
    0: "angry.png",       # Angry
    1: "disgusted.png",   # Disgusted (if missing, will fall back)
    2: "fearful.png",     # Fearful (if missing, will fall back)
    3: "happy.png",       # Happy (if missing, will fall back)
    4: "neutral.png",     # Neutral
    5: "sad.png",         # Sad
    6: "surpriced.png",   # Surprised 
}

def get_emoji_path(emotion_index: int):
    """Return path to emoji image; fall back to neutral if missing. Searches known emoji dirs."""
    filename = emoji_filenames.get(emotion_index, "neutral.png")
    for base in possible_emojis_dirs:
        candidate = base / filename
        if candidate.exists():
            return str(candidate)
    # Fallback to neutral wherever it exists
    for base in possible_emojis_dirs:
        fallback = base / "neutral.png"
        if fallback.exists():
            return str(fallback)
    print(f"[warning] Emoji image not found for index {emotion_index}: {filename}. Checked: {possible_emojis_dirs}")
    return None

def open_camera():
    # Try common options on Windows
    for cam_index in (0, 1, 2):
        cap = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW)
        if cap.isOpened():
            return cap
        cap.release()
        cap = cv2.VideoCapture(cam_index)
        if cap.isOpened():
            return cap
        cap.release()
    return cv2.VideoCapture(0)

cap = open_camera()
show_text = [0]

def show_vid():
    ret, frame = cap.read()
    if not ret:
        return

    frame = cv2.resize(frame, (600, 500))
    # Mirror the camera view horizontally (front-facing camera behavior)
    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = face_detector.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in faces:
        roi = gray[y:y+h, x:x+w]
        roi = cv2.resize(roi, (48,48))
        roi = roi.reshape(1,48,48,1)
        prediction = emotion_model.predict(roi)
        show_text[0] = np.argmax(prediction)

    img = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
    lmain.imgtk = img
    lmain.configure(image=img)
    lmain.after(10, show_vid)

def show_vid2():
    path = get_emoji_path(show_text[0])
    if path:
        frame2 = cv2.imread(path)
        if frame2 is not None:
            # Resize emoji to a consistent size so it is visible
            frame2 = cv2.resize(frame2, (400, 400))
            img2 = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)))
            lmain2.imgtk2 = img2
            lmain2.configure(image=img2)
        else:
            # If reading failed, clear image area
            lmain2.configure(image="")
    else:
        # No valid path: clear image area
        lmain2.configure(image="")
    # Always update label text
    lmain3.configure(text=emotion_dict.get(show_text[0], ""), font=('arial', 45, 'bold'))
    lmain2.after(10, show_vid2)

root = tk.Tk()
root.title("Photo To Emoji")
root.geometry("1400x900+100+10")
root.configure(bg='black')

heading2 = Label(root, text="Photo to Emoji", pady=20, font=('arial', 45, 'bold'), bg='black', fg='#CDCDCD')
heading2.pack()

lmain = tk.Label(root, bd=10, bg='black')
lmain.place(x=50, y=250)

lmain3 = tk.Label(root, bd=10, fg="#CDCDCD", bg='black')
lmain3.place(x=900, y=250)

lmain2 = tk.Label(root, bd=10, bg='black')
lmain2.place(x=900, y=350)

Button(root, text='Quit', fg="red", command=root.destroy, font=('arial', 25, 'bold')).pack(side=BOTTOM)

show_vid()
show_vid2()
root.mainloop()
