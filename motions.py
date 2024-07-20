import tensorflow as tf
import cv2
import mediapipe as mp
import numpy as np

from collections import deque
from dataclasses import dataclass
from sklearn.discriminant_analysis import StandardScaler
from time import sleep, time
from threading import Thread, Event

from helpers import landmarks_from_index, prepare_global_scaler

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode


@dataclass
class HandLandmarks:
    left_hand: deque[list[tuple[float, float]]]
    right_hand: deque[list[tuple[float, float]]]

    @property
    def scalled_left_hand(self) -> np.ndarray:
        scaller = StandardScaler()
        scaller.fit(self.left_hand[0])

        return np.array(list(map(scaller.transform, self.left_hand)))

    @property
    def scalled_right_hand(self) -> np.ndarray:
        scaller = StandardScaler()
        scaller.fit(self.right_hand[0])

        return np.array(list(map(scaller.transform, self.right_hand)))


class HandMotions:
    def __init__(self, camera_id: int, model_names: tuple[str, str], global_scaller: StandardScaler, global_hands: HandLandmarks) -> None:
        self.global_scaller = global_scaller

        self._camera = cv2.VideoCapture(camera_id)
        self._model_left = tf.keras.models.load_model(model_names[0])
        self._model_right = tf.keras.models.load_model(model_names[1])

        # FPS
        self.fps = 20
        self.frame_duration = 1/self.fps

        # Hand values
        self.global_hands = global_hands

        # Hand model options
        def result_callback(result, output_image, timestamp_ms) -> None:
            for category in result.handedness:
                if category[0].category_name == "Left":
                    global_hands.left_hand.append(list(
                        (lm.x, lm.y) for lm in landmarks_from_index(category[0].index, result.hand_landmarks)
                    ))
                elif category[0].category_name == "Right":
                    global_hands.right_hand.append(list(
                        (lm.x, lm.y) for lm in landmarks_from_index(category[0].index, result.hand_landmarks)
                    ))
        self._landmarker_options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path='hand_landmarker.task'),
            num_hands=2,
            running_mode=VisionRunningMode.LIVE_STREAM,
            result_callback=result_callback,
        )

        # Threads
        self._stop_event = Event()
        self._camera_thread = Thread(target=self._capture_loop)

    def _capture_loop(self) -> None:
        """
            Only to use inside of a Thread

            Captures hand landmarks 20 times per second

            https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker/python?hl=en
        """
        with HandLandmarker.create_from_options(self._landmarker_options) as landmarker:
            while not self._stop_event.is_set():
                start = time()

                # Frame processing
                _, frame = self._camera.read()
                mp_image = mp.Image(
                    image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

                # Detecting landmarks
                landmarker.detect_async(mp_image, int(start*1000))

                # Limit FPS
                if (time_to_sleep := self.frame_duration - time() + start) > 0:
                    sleep(time_to_sleep)
                # print(self.global_hands.right_hand)

    @property
    def status_left(self) -> int:
        return np.argmax(self._model_left(self.global_scaller.transform([self.global_hands.scalled_left_hand.flatten()]).reshape(1, -1), training=False))

    @property
    def status_right(self) -> int:
        return np.argmax(self._model_right(self.global_scaller.transform([self.global_hands.scalled_right_hand.flatten()]).reshape(1, -1), training=False))

    def start_camera_loop(self) -> None:
        self._camera_thread.start()

    def stop_camera_loop(self) -> None:
        self._stop_event.set()
        self._camera_thread.join()


if __name__ == "__main__":
    from os import system

    system("cls")
    global_hand = HandLandmarks(
        left_hand=deque([[(0.0, 0.0) for _ in range(21)]
                        for _ in range(20)], maxlen=20),
        right_hand=deque([[(0.0, 0.0) for _ in range(21)]
                         for _ in range(20)], maxlen=20)
    )

    hand_motions = HandMotions(0, ("model.h5", "model.h5"), prepare_global_scaler(), global_hand)
    hand_motions.start_camera_loop()

    while True:
        print(hand_motions.status_right)
        sleep(1)
