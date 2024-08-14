from collections import deque
from time import sleep
from os import system
import warnings

from motions import HandLandmarks, HandMotions
from helpers import SETTINGS, prepare_global_scaler
from riot_api import Game
from mouse_control import MouseController

FPS = SETTINGS["program"]["fps"]

if __name__ == "__main__":
    # Initialize global hand dataclass
    global_hand = HandLandmarks(
        left_hand=deque([[(0.0, 0.0, 0.0) for _ in range(21)]
                        for _ in range(20)], maxlen=20),
        right_hand=deque([[(0.0, 0.0, 0.0) for _ in range(21)]
                         for _ in range(20)], maxlen=20)
    )

    # Initialize classes
    # api = Game(SETTINGS["lor"]["port"])
    hand_motions = HandMotions(
        0, ("models/lhm3d.keras", "models/rhm3d.h5"), prepare_global_scaler(), global_hand)
    controller = MouseController(SETTINGS["screen"])

    # Start loops
    # api.start()
    hand_motions.start_camera_loop()
    
    with warnings.catch_warnings(action="ignore"):
        while True: 
            system("cls")
            status = hand_motions.status
            print(status)

            # Calibrating
            if status.probable_left == 1:
                controller.calibrate(global_hand.right_hand[-1])
                print(controller.center, controller.hand_constant)

            # Game control
            if status.probable_right == 1:
                controller.next_turn()

            elif status.probable_right == 2:
                print(controller._get_current_selected_card(5, global_hand.pointed_fingers))


            sleep(.5)
