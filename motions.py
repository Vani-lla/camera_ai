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

from helpers import SETTINGS, normalized_to_px, prepare_global_scaler


@dataclass
class HandLandmarks:
    left_hand: deque[list[tuple[float, float, float]]]
    right_hand: deque[list[tuple[float, float, float]]]

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

    @property
    def pointed_fingers(self) -> float:
        return (self.right_hand[-1][8][0] + self.right_hand[-1][12][0])/2


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
    def __init__(self, camera_id: int, model_names: tuple[str, str] | None, global_scaller: StandardScaler | None, global_hands: HandLandmarks) -> None:
        if global_scaller is not None:
            self.global_scaller = global_scaller

        self._camera = cv2.VideoCapture(camera_id)
        if model_names is not None:
            self._model_left = tf.keras.models.load_model(model_names[0])
            self._model_right = tf.keras.models.load_model(model_names[1])

        # FPS
        self.fps = 20
        self.frame_duration = 1/self.fps

        # Hand values
        self.global_hands = global_hands

        # Hand model options
        BaseOptions = mp.tasks.BaseOptions
        HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        def result_callback(result, output_image, timestamp_ms) -> None:
            for hand_landmarks, handedness in zip(result.hand_landmarks, result.handedness):
                category = handedness[0]

                if category.category_name == "Left":
                    global_hands.left_hand.append(list(
                        (lm.x, lm.y, lm.z) for lm in hand_landmarks
                    ))
                elif category.category_name == "Right":
                    global_hands.right_hand.append(list(
                        (lm.x, lm.y, lm.z) for lm in hand_landmarks
                    ))
            return timestamp_ms

        self._landmarker_options = HandLandmarkerOptions(
            base_options=BaseOptions(
                model_asset_path="models/hand_landmarker.task"),
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
        HandLandmarker = mp.tasks.vision.HandLandmarker
        with HandLandmarker.create_from_options(self._landmarker_options) as landmarker:
            while not self._stop_event.is_set():
                start = time()

                # Frame processing
                _, frame = self._camera.read()

                cv2.imshow("camera", frame)
                cv2.waitKey(5)

                mp_image = mp.Image(
                    image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

                # Detecting landmarks
                landmarker.detect_async(mp_image, int(start*1000))

                # Limit FPS
                if (time_to_sleep := self.frame_duration - time() + start) > 0:
                    sleep(time_to_sleep)

    @property
    def status(self) -> Status:
        data_left = np.array(self.global_hands.left_hand[-1]).reshape(-1, 63)
        data_right = self.global_scaller.transform(
            [self.global_hands.scalled_right_hand.flatten()]).reshape(1, -1)

        left_prediction = self._model_left(data_left, training=False)
        right_prediction = self._model_right(data_right, training=False)

        left_status = np.argmax(left_prediction)
        right_status = np.argmax(right_prediction)

        return Status(left_status, right_status, left_prediction[0, left_status], right_prediction[0, right_status])

    def start_camera_loop(self) -> None:
        self._camera_thread.start()

    def stop_camera_loop(self) -> None:
        self._stop_event.set()
        self._camera_thread.join()

# TODO: Make it better
# class SingularMotionRecorder:
#     def __init__(self, camera_id: int, motion_id: int, singular: bool, single_frame_motion: bool, hand: Literal["Left", "Right"], global_deque: deque[list[float, float, float]]) -> None:
#         self.motion_id = motion_id
#         self.hand = hand

#         self.singular = singular
#         self.single_frame_motion = single_frame_motion

#         self._camera = cv2.VideoCapture(camera_id)

#         # Hand model options
#         BaseOptions = mp.tasks.BaseOptions
#         HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
#         VisionRunningMode = mp.tasks.vision.RunningMode

#         def result_callback(result, output_image, timestamp_ms) -> None:
#             for category in result.handedness:
#                 if category[0].category_name == hand:
#                     global_deque.append(list(
#                         (lm.x, lm.y, lm.z) for lm in landmarks_from_index(category[0].index, result.hand_landmarks)
#                     ))

#         self._landmarker_options = HandLandmarkerOptions(
#             base_options=BaseOptions(
#                 model_asset_path="models/hand_landmarker.task"),
#             num_hands=2,
#             running_mode=VisionRunningMode.LIVE_STREAM,
#             result_callback=result_callback,
#         )
#         # 20 x 21 x 3
#         self.global_deque = global_deque
#         self._motions = []

#     def _record_motion(self) -> None:
#         scaller = StandardScaler()
#         scaller.fit(self.global_deque[0])

#         try:
#             self._motions.append(
#                 list(map(scaller.transform, self.global_deque)))
#         except RuntimeError:
#             return

#     def _save_file(self) -> None:
#         np.save(
#             f"training_data/{self.hand.lower()}/motion_{self.motion_id}", np.array(self._motions))

#     def recording_loop(self) -> None:
#         import pygame
#         pygame.init()

#         WIDTH, HEIGHT = 1280, 720
#         win = pygame.display.set_mode((WIDTH, HEIGHT))
#         clock = pygame.time.Clock()

#         recording = -1
#         run = True
#         HandLandmarker = mp.tasks.vision.HandLandmarker
#         with HandLandmarker.create_from_options(self._landmarker_options) as landmarker:
#             while run:
#                 # Frame processing
#                 _, frame = self._camera.read()
#                 mp_image = mp.Image(
#                     image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

#                 # Detecting landmarks
#                 landmarker.detect_async(mp_image, int(time()*1000))

#                 # Pygame events
#                 for event in pygame.event.get():
#                     if event.type == pygame.QUIT:
#                         run = False
#                     if event.type == pygame.KEYDOWN:
#                         if event.key == pygame.K_SPACE and recording == -1:
#                             recording = 20

#                 # Fill screen
#                 if recording > 0:
#                     win.fill((55, 155, 55))
#                 else:
#                     win.fill((0, 0, 0))

#                 # Display landmarks
#                 for x, y, _ in self.global_deque[-1]:
#                     x_, y_ = np.array(normalized_to_px(
#                         x, y, width=WIDTH, height=HEIGHT))//3
#                     pygame.draw.circle(win, (255, 255, 255),
#                                        (x_+WIDTH//2, y_+HEIGHT//2), 5)
#                 pygame.display.update()

#                 # Show camera
#                 cv2.imshow("x", frame)
#                 if cv2.waitKey(1) == ord("q"):
#                     run = False

#                 # Save motion
#                 if self.singular:
#                     if recording == 0:
#                         self._record_motion()
#                     if recording >= 0:
#                         recording -= 1
#                 elif not self.single_frame_motion:
#                     self._record_motion()
#                 else:
#                     self._motions.append(
#                         StandardScaler().fit_transform(self.global_deque[-1]))

#                 print(clock.get_fps())
#                 clock.tick(20)
#         pygame.quit()

#         self._save_file()


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
        0, ("models/lhm3d.keras", "models/rhm3d.h5"), prepare_global_scaler(), global_hand)
    hand_motions.start_camera_loop()

    while True:
        sleep(1)
        print(hand_motions.status)

    # x = deque([[(0.0, 0.0, 0.0) for _ in range(21)] for _ in range(20)], maxlen=20)

    # recorder = SingularMotionRecorder(0, -1, False, True, "Left", x)
    # recorder.recording_loop()
