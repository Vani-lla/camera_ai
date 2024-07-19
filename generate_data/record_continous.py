import numpy as np
import cv2
import mediapipe.python.solutions.hands as hands
import pygame
from os import system

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

import warnings
system("clear")
warnings.filterwarnings("ignore")

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
    
    tmp_data = np.array(list(scaller.transform(row) for row in data))
    
    # tmp_data = np.array(list([tmp_data]))
    
    return tmp_data

run = True
data = []
total_data = []
landmarks = []
# Main loop
while run:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            run = False
            
    success, frame_orig = cap.read()
    frame = cv2.cvtColor(frame_orig, cv2.COLOR_BGR2RGB)
    result = hand.process(frame)
    
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks[:1]:
            landmarks = np.array(list(map(lambda lm: (lm.x, lm.y), hand_landmarks.landmark)))
            
            data.append(landmarks)
            if len(data) > 20:
                data.pop(0)
                
                total_data.append(get_scalled_data(data))
                
    elif len(landmarks) > 0:
        data.append(landmarks)
        if len(data) > 20:
            data.pop(0)
            
            total_data.append(get_scalled_data(data))
    
    cv2.imshow("x", frame_orig)
    if cv2.waitKey(1) == ord("q"):
        run = False
        
    clock.tick(20)

total_data = np.array(total_data)
print(total_data.shape)
np.save("training_data/4.npy", total_data)

pygame.quit()