from collections import deque
import numpy as np
import pygame

from helpers import normalized_to_px, prepare_global_scaler
from motions import HandMotions, HandLandmarks
from sklearn.preprocessing import StandardScaler

pygame.init()

WIDTH, HEIGHT = 1280, 720
win = pygame.display.set_mode((WIDTH, HEIGHT))

global_hands = HandLandmarks(
    left_hand=deque([[(0.0, 0.0) for _ in range(21)]
                     for _ in range(20)], maxlen=20),
    right_hand=deque([[(0.0, 0.0) for _ in range(21)]
                      for _ in range(20)], maxlen=20)
)
hand_motions = HandMotions(
    0, ("models/model.h5", "models/model.h5"), prepare_global_scaler(), global_hands)
hand_motions.start_camera_loop()

run = True
while run:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            run = False
            hand_motions.stop_camera_loop()

    win.fill((0, 0, 0))
    hand = StandardScaler().fit_transform(global_hands.right_hand[-1])
    for x, y in hand:
        x_, y_ = np.array(normalized_to_px(
            x, y, width=WIDTH, height=HEIGHT))//10

        pygame.draw.circle(win, (255, 255, 255),
                           (x_+WIDTH//2, y_+HEIGHT//2), 5)

    pygame.display.update()

pygame.quit()
