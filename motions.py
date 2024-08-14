from typing import Literal
import tensorflow as tf
import cv2
import mediapipe as mp
import numpy as np

from collections import deque
from dataclasses import dataclass
from sklearn.discriminant_analysis import StandardScaler
from time import sleep, time
from threading import Thread, Event

from recorder import HandLandmarks
from helpers import SETTINGS, normalized_to_px


@dataclass
class Status:
    left: int = 0
    right: int = 0

    left_probability: float = 1
    right_probability: float = 1

    @property
    def probable_left(self) -> int:
        return self.left if self.left_probability > SETTINGS["program"]["confidence"] else 0

    @property
    def probable_right(self) -> int:
        return self.right if self.right_probability > SETTINGS["program"]["confidence"] else 0

    def __repr__(self) -> str:
        return f"Left: {self.left} {self.left_probability}\t\t\tRight: {self.right} {self.right_probability}"

class HandMotions:
    def __init__(self, model_names: tuple[str, str], global_hands) -> None:
        self._model_left = tf.keras.models.load_model(model_names[0])
        self._model_right = tf.keras.models.load_model(model_names[1])
        
        self.global_hands = global_hands

    def generate_status(self) -> Status:
        data_left = np.array(self.global_hands.left_hand[-1]).reshape(-1, 63)
        data_right = self.global_hands.scalled_right_hand.flatten().reshape(1, -1)

        left_prediction = self._model_left(data_left, training=False)
        right_prediction = self._model_right(data_right, training=False)

        left_status = np.argmax(left_prediction)
        right_status = np.argmax(right_prediction)

        return Status(left_status, right_status, left_prediction[0, left_status], right_prediction[0, right_status])


if __name__ == "__main__":
    from os import system
    from warnings import filterwarnings
    filterwarnings("ignore")

    global_hand = HandLandmarks(
        left_hand=deque([[(0.0, 0.0, 0.0) for _ in range(21)]
                        for _ in range(20)], maxlen=20),
        right_hand=deque([[(0.0, 0.0, 0.0) for _ in range(21)]
                         for _ in range(20)], maxlen=20)
    )
    hand_motions = HandMotions(
        0, ("models/lhm3d.keras", "models/rhm3d.h5"), global_hand)
    hand_motions.start_camera_loop()

    while True:
        sleep(1)
        print(hand_motions.generate_status)

    # x = deque([[(0.0, 0.0, 0.0) for _ in range(21)] for _ in range(20)], maxlen=20)

    # recorder = SingularMotionRecorder(0, -1, False, True, "Left", x)
    # recorder.recording_loop()
