import cv2 as cv # computer vison to read data from camera
import numpy as np 
import mediapipe as mp
import os
import sklearn
import time
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split

# --- CONFIGURATION ---
# Define the specific signs (classes) we will train the ANN on (26 Letters + 5 Phrases)
# NOTE: ASL uses single handshapes for many of these, making them suitable for this model.
#Letter and Phrase Classes must have all labels to be trained
LETTER_CLASSES = [
    'A', 'B', 'C', 'D', 'E'
]
PHRASE_CLASSES = [
    'THANK_YOU', 'PLEASE', 'YES', 'NO', 'STOP' 
    # Note: 'HOW_ARE_YOU' is often a single, fluid sign or short sequence,
    # but we will train the final handshape/frame for simplicity here.
]

SIGN_CLASSES = LETTER_CLASSES + PHRASE_CLASSES

# Increased samples per sign for 31 total classes
SAMPLES_PER_SIGN = 160 # Number of hand poses (samples) to capture for each sign

# --- INITIALIZATION ---
# Initializing the MediaPipe Hands Model
mp_hands = mp.solutions.hands
# Use static_image_mode=False for video stream
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

# Video Capture Setup
cap = cv.VideoCapture(0, cv.CAP_DSHOW) 
cap.set(cv.CAP_PROP_FOURCC, cv.VideoWriter_fourcc(*'MJPG'))
cap.set(cv.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, 720)

# The Neural Network Classifier (will be trained after data collection)
classifier = None 

# --- PHASE 1: DATA COLLECTION AND FEATURE EXTRACTION ---

def extract_features(results):
    """
    Extracts the 63-dimensional feature vector (x, y, z for 21 landmarks).
    Returns None if no hand is detected.
    """
    if results.multi_hand_landmarks:
        # We process the first detected hand
        handLms = results.multi_hand_landmarks[0]
        keypoints = []
        # Extract x, y, z coordinates for all 21 landmarks
        for lm in handLms.landmark:
            # Normalized coordinates (0.0 to 1.0)
            keypoints.extend([lm.x, lm.y, lm.z])
        # Return the 63-dimensional feature vector
        return np.array(keypoints) 
    return None

def process_frame_and_extract(frame, hands_model, mp_draw, draw=True):
    """Detects hand points and extracts features."""
    imgRGB = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    imgRGB.flags.writeable = False # Performance boost
    results = hands_model.process(imgRGB)
    imgRGB.flags.writeable = True

    features = extract_features(results)

    if results.multi_hand_landmarks and draw:
        # Draw landmarks on the frame
        mp_draw.draw_landmarks(frame, results.multi_hand_landmarks[0], 
                               mp_hands.HAND_CONNECTIONS)
        
    return frame, features

def collect_training_data():
    """
    Collects real-time hand pose data from the camera for each defined sign.
    """
    X = [] # Features (63 dimensions per sample)
    y = [] # Labels (class index)
    
    print(f"\n--- STARTING DATA COLLECTION FOR {len(SIGN_CLASSES)} SIGNS ---")
    
    for class_index, sign in enumerate(SIGN_CLASSES):
        # Time delay to allow user to get ready
        print(f"\nReady to capture sign: '{sign}'. Get your hand into position!")
        time.sleep(3) 
        
        captured_count = 0
        while captured_count < SAMPLES_PER_SIGN:
            ret, frame = cap.read()
            if not ret: break

            frame = cv.flip(frame, 1) # Mirror image for intuitive control
            frame, features = process_frame_and_extract(frame, hands, mp_draw, draw=True)
            
            if features is not None:
                # Store the feature vector and its label
                X.append(features)
                y.append(class_index)
                captured_count += 1
            
            # Display current status in the window
            cv.putText(frame, f"Sign: '{sign}' | Samples: ({captured_count}/{SAMPLES_PER_SIGN})", 
                       (50, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv.LINE_AA)
            cv.imshow("Live Sign Recognition - Data Collection", frame) 
            
            if cv.waitKey(5) & 0xFF == ord('q'):
                break
        
        print(f"Captured {captured_count} samples for '{sign}'.")
        
    cv.destroyWindow("Live Sign Recognition - Data Collection")
    print("\n--- DATA COLLECTION COMPLETE ---")
    return np.array(X), np.array(y)

# --- PHASE 2: TRAIN ARTIFICIAL NEURAL NETWORK (ANN) ---

def train_ann_model(X, y):
    """
    Trains the Multilayer Perceptron (ANN) classifier.
    """
    if X.shape[0] < SAMPLES_PER_SIGN * len(SIGN_CLASSES) / 2:
        print("ERROR: Insufficient data captured for reliable training.")
        return None
        
    # Split the data into training (90%) and testing (10%) sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

    # Initialize the MLPClassifier (our simple ANN)
    # The network must distinguish between 31 classes.
    model = MLPClassifier(hidden_layer_sizes=(100, 50), 
                          max_iter=600, # Increased max_iter for more training time
                          activation='relu', 
                          solver='adam', 
                          verbose=True, 
                          random_state=1)
    
    print("\n--- TRAINING NEURAL NETWORK (ANN) ---")
    model.fit(X_train, y_train) 
    print("--- TRAINING COMPLETE ---")

    # Evaluate the model's accuracy
    accuracy = model.score(X_test, y_test)
    print(f"ANN Model Test Accuracy on unseen data: {accuracy:.2f}")

    return model

# --- MAIN EXECUTION LOGIC ---

# 1. Collect Data
X_data, y_labels = collect_training_data()

# 2. Train Model
if X_data.size > 0:
    classifier = train_ann_model(X_data, y_labels)
else:
    print("No data collected. Exiting.")
    cap.release()
    cv.destroyAllWindows()
    exit()

# 3. Live Prediction Loop (PHASE 3)

current_sign_text = "Ready to Predict!"
prediction_start_time = time.time()
PREDICTION_DISPLAY_DURATION = 1.5 # Show prediction for 1.5 seconds

print("\n--- STARTING LIVE PREDICTION ---")

while True:
    ret, frame = cap.read()
    if not ret: 
        print("Camera stream ended.")
        break
    
    frame = cv.flip(frame, 1) # Mirror image

    # Get new features from the live camera feed
    frame, features = process_frame_and_extract(frame, hands, mp_draw, draw=True)

    if features is not None and classifier is not None:
        try:
            features = features.reshape(1, -1)
            # Pass the features to the trained ANN for classification
            prediction_index = classifier.predict(features)[0]
            
            # Look up the sign and update display variables
            predicted_sign = SIGN_CLASSES[prediction_index]
            
            current_sign_text = f"Predicted: {predicted_sign}"
            prediction_start_time = time.time()

        except Exception as e:
            current_sign_text = "Prediction Error"
            
    # Display the result
    current_time = time.time()
    
    display_text = current_sign_text
    # Check if the prediction is recent
    if (current_time - prediction_start_time) > PREDICTION_DISPLAY_DURATION:
        display_text = "Show a sign..."

    cv.putText(frame, display_text, (50, 50), 
               cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv.LINE_AA)

    cv.imshow("Live Sign Recognition (ANN)", frame)
    
    if cv.waitKey(1) == ord("q"):
        break

# --- CLEANUP ---
cap.release()
cv.destroyAllWindows()
print("Application closed.")