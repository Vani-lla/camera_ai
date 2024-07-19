import cv2
import mediapipe.python.solutions.hands as hands
import numpy as np
import pygame
from sklearn.preprocessing import StandardScaler

import warnings
warnings.filterwarnings("ignore")

pygame.init()

WIDTH, HEIGHT = 1280, 720
win = pygame.display.set_mode((WIDTH, HEIGHT))
clock = pygame.time.Clock()

cap = cv2.VideoCapture(0)
mp_hands = hands
hand = mp_hands.Hands(max_num_hands=2, static_image_mode=False, model_complexity=1, )

data = []
total_data = []

recording = 0
run = True
while run:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            run = False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE and recording == 0:
                recording = 20
                print("X")
            
    success, frame_orig = cap.read()
    frame = cv2.cvtColor(frame_orig, cv2.COLOR_BGR2RGB)
    result = hand.process(frame)

    positions = []
    if recording == 20:
        try:
            for hand_landmarks in result.multi_hand_landmarks:
                landmarks = np.array(list(map(lambda lm: (lm.x, lm.y), hand_landmarks.landmark)))
                
                scaller = StandardScaler()
                scaller.fit(landmarks)
        except:
            recording = 0
                
    if recording > 0:
        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks[:1]:
                landmarks = np.array(list(map(lambda lm: (lm.x, lm.y), hand_landmarks.landmark)))
                landmarks = scaller.transform(landmarks)
                
                data.append(landmarks)
                # print(recording)
                
                landmarks /= 10
                for x, y in landmarks:
                    x = int(WIDTH*(x + 0.5))
                    y = int(HEIGHT*(y + 0.5))
                    
                    positions.append((x, y))
        else:
            data.append(landmarks)
            
        recording -= 1
    
    if recording > 0:
        win.fill((55, 155, 55))
    else:
        if len(data) > 0:
            print(np.array(data).shape)
            total_data.append(data)
            print(f"Saved {len(total_data)}")
            data = []
        win.fill((0, 0, 0))
        
    for x, y in positions:
        pygame.draw.circle(win, (255, 255, 255), (x, y), 5)
    pygame.display.update()
        
    cv2.imshow("x", frame_orig)
    if cv2.waitKey(1) == ord("q"):
        run = False
    
    clock.tick(20)
    # print(int(clock.get_fps()), recording)
    
print(total_data)
total_data = np.array(total_data)
print(total_data.shape)
np.save("nothing", total_data)
pygame.quit()