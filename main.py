import numpy as np
import cv2
import mediapipe.python.solutions.hands as hands
import pygame
import pyautogui
import tensorflow as tf

from os import system

from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

import warnings


system("clear")
warnings.filterwarnings("ignore")

# Setting up model
WIDTH = 1920

X, Y = [], []
for i in range(5):
    tmp = np.load(f"training_data/{i}.npy")
    X.append(tmp)
    for _ in range(len(tmp)):
        Y.append(i)

X = np.row_stack(X).reshape(-1, 21*2*20)
Y = np.array(Y)

global_scaller = StandardScaler()
# lda = LinearDiscriminantAnalysis()
X = global_scaller.fit_transform(X)
# X = lda.fit_transform(X, Y)

model = LogisticRegression()
# model = KNeighborsClassifier()
# model = DecisionTreeClassifier()
# model = SVC()
ai = tf.keras.models.load_model("model.h5")
model.fit(X, Y)

# Pygame Initialization
pygame.init()
clock = pygame.time.Clock()

# Mediapipe hands model
cap = cv2.VideoCapture(0)
mp_hands = hands
hand = mp_hands.Hands(max_num_hands=2, static_image_mode=False, model_complexity=1, )

# Defining variables and key functions
def get_scalled_data(data: list) -> np.ndarray:
    scaller = StandardScaler()
    scaller.fit(data[0])
    
    tmp_data = np.array(list(scaller.transform(row) for row in data)).flatten()
    
    tmp_data = np.array(list([tmp_data]))
    
    return global_scaller.transform(tmp_data).reshape(1, -1)
    # return lda.transform(global_scaller.transform(tmp_data).reshape(1, -1))

run = True
data, statuses = [], []
prev, center = 0, WIDTH//2
card_y = 500
hand_constant = 0
# Main loop
while run:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            run = False
            
    success, frame_orig = cap.read()
    frame = cv2.cvtColor(frame_orig, cv2.COLOR_BGR2RGB)
    result = hand.process(frame)
    
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            landmarks = np.array(list(map(lambda lm: (lm.x, lm.y), hand_landmarks.landmark)))
            
            data.append(landmarks)
            if len(data) > 20:
                data.pop(0)
    
    cv2.imshow("x", frame_orig)
    
    system("clear")
    key = cv2.waitKey(1)
    if key == ord("q"):
        run = False
    elif key == ord("e"):
        hand_constant = landmarks[4][0] - landmarks[20][0]
        print(hand_constant)
        
    if len(data) == 20:
        # statuses.append(model.predict(get_scalled_data(data)))
        # if len(statuses) == 5:
            # statuses.pop(0)
        # status = max(statuses, key=statuses.count)
        # status = statuses[-1]

        # if status == 2:
        #     # finger = (landmarks[8][0] + landmarks[12][0])/2
        #     finger = np.average(list(map(lambda x: x[0], landmarks)))
        #     if prev != 2:
        #         center = (landmarks[8][0] + landmarks[12][0])/2

        #     else:
        #         pos = (finger - .5*center)/hand_constant
        #         print(pos)
        #         pyautogui.moveTo(int(pos*WIDTH), card_y, _pause=False)
        # system("clear")
        ai_status = np.argmax(ai(get_scalled_data(data), training=False), axis=1)
        # print(f"FPS: {int(clock.get_fps())}\t Status: {status}\t Center: {center}")
        print(f"FPS: {int(clock.get_fps())}\t AI: {ai_status}")
        # prev = status
        # print(ai_status)
        
    clock.tick()
        
pygame.quit()