import cv2
import time
import mediapipe as mp

# Function to recognize hand gestures
def recognize_hand_gesture(hand_landmarks):
    # Counting the number of fingers raised
    num_fingers_raised = 0
    
    # Detecting thumb (if thumb tip above index finger tip)
    if hand_landmarks.landmark[mp.solutions.hands.HandLandmark.THUMB_TIP].y < hand_landmarks.landmark[mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP].y:
        num_fingers_raised += 1
    
    # Detecting index, middle, ring, and pinky fingers (if respective tips above respective bases)
    for landmark in [mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP, 
                     mp.solutions.hands.HandLandmark.MIDDLE_FINGER_TIP,
                     mp.solutions.hands.HandLandmark.RING_FINGER_TIP,
                     mp.solutions.hands.HandLandmark.PINKY_TIP]:
        if hand_landmarks.landmark[landmark].y < hand_landmarks.landmark[mp.solutions.hands.HandLandmark.WRIST].y:
            num_fingers_raised += 1
    
    # Classifying gestures based on the number of fingers raised
    if num_fingers_raised == 0:
        return "Fist"
    elif num_fingers_raised == 1:
        return "Hand Raised"
    elif num_fingers_raised == 2:
        return "Hand Raised"
    elif num_fingers_raised == 3:
        return "Hand Raised"
    elif num_fingers_raised == 4:
        return "Hand Raised"
    else:
        return "Palm"

# Grabbing the Hand Model from Mediapipe and Initializing the Model
mp_hands = mp.solutions.hands
hands_model = mp_hands.Hands(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Initializing the drawing utils for drawing the hand landmarks on image
mp_drawing = mp.solutions.drawing_utils

# (0) in VideoCapture is used to connect to your computer's default camera
capture = cv2.VideoCapture(0)

# Initializing current time and previous time for calculating the FPS
previousTime = 0
currentTime = 0

while capture.isOpened():
    # Capture frame by frame
    ret, frame = capture.read()

    # Resizing the frame for better view
    frame = cv2.resize(frame, (800, 600))

    # Converting the from BGR to RGB
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Making predictions using hands model
    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    results = hands_model.process(image)
    image.flags.writeable = True

    # Converting back the RGB image to BGR
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Drawing the Hand Landmarks
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                image,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )
            
            # Recognizing hand gestures
            gesture = recognize_hand_gesture(hand_landmarks)
            cv2.putText(image, gesture, (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Calculating the FPS
    currentTime = time.time()
    fps = 1 / (currentTime - previousTime)
    previousTime = currentTime

    # Displaying FPS on the image
    cv2.putText(image, f"{int(fps)} FPS", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the resulting image
    cv2.imshow("Hand Gesture Recognition", image)

    # Enter 'q' to break the loop
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

# Release the capture and destroy all windows
capture.release()
cv2.destroyAllWindows()
