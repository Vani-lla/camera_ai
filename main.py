from collections import deque
from json import load
from time import sleep

from motions import HandLandmarks, HandMotions
from helpers import prepare_global_scaler
from riot_api import Game
from mouse_control import MouseController

with open("settings/settings.json") as file:
    SETTINGS: dict = load(file)
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
    api = Game(SETTINGS["lor"]["port"])
    hand_motions = HandMotions(
        0, ("models/lhm3d.h5", "models/rhm3d.h5"), prepare_global_scaler(), global_hand)
    controller = MouseController(SETTINGS["screen"])

    # Start loops
    api.start()
    hand_motions.start_camera_loop()

    while True:
        controller.status = hand_motions.status
        