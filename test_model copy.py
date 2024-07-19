import cv2
import mediapipe.python.solutions.hands as hands
import numpy as np
import pygame

pygame.init()

WIDTH, HEIGHT = 1280, 720
win = pygame.display.set_mode((WIDTH, HEIGHT))

cap = cv2.VideoCapture(0)
mp_hands = hands
hand = mp_hands.Hands(max_num_hands=2, static_image_mode=False, model_complexity=1, )

run = True
# positions = [Queue(20) for _ in range(21)]
positions = []
while run:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            run = False
    
    success, frame_orig = cap.read()
    frame = cv2.cvtColor(frame_orig, cv2.COLOR_BGR2RGB)
    result = hand.process(frame)
    
    positions = []
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # x = np.average(list(map(lambda lm: lm.x, hand_landmarks.landmark)))
            # y = np.average(list(map(lambda lm: lm.y, hand_landmarks.landmark)))
            for lm in hand_landmarks.landmark:
                x = lm.x
                y = lm.y
                
                x = int(WIDTH*(1-x))
                y = int(HEIGHT*y)
                
                positions.append((x, y))
    
    win.fill((0, 0, 0))
    for x, y in positions:
        pygame.draw.circle(win, (255, 100, 100), (x, y), 5)
    pygame.display.update()
    
    if cv2.waitKey(1) == ord("q"):
        run = False
        
pygame.quit()