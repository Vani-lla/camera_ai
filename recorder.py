from collections import deque
from threading import Event, Thread
from time import sleep, time
from typing import Literal
import cv2
import numpy as np
from sklearn.discriminant_analysis import StandardScaler
from dataclasses import dataclass
import mediapipe as mp

from helpers import camera_loop_tick, normalized_to_px


@dataclass
class HandLandmarks:
    left_hand: deque[list[tuple[float, float, float]]]
    right_hand: deque[list[tuple[float, float, float]]]

    def get_hand_from_str(self, hand: str) -> deque[list[tuple[float, float, float]]]:
        return getattr(self, f"{hand.lower()}_hand")

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


class LandmarkRecorder:
    def __init__(self, global_hands: HandLandmarks, show_camera: bool, camera_id: int = 0) -> None:
        # Camera
        self._camera = cv2.VideoCapture(camera_id)
        self.show_camera = show_camera

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
                self.global_hands.get_hand_from_str(category.category_name).append(list(
                    (lm.x, lm.y, lm.z) for lm in hand_landmarks
                ))

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
            if self.show_camera:
                while not self._stop_event.is_set():
                    time_taken, frame = camera_loop_tick(
                        self._camera, landmarker)

                    # Show frame
                    cv2.imshow(frame)
                    cv2.waitKey(5)

                    # Limit FPS
                    if (time_to_sleep := self.frame_duration - time_taken) > 0:
                        sleep(time_to_sleep)
            else:
                while not self._stop_event.is_set():
                    time_taken, _ = camera_loop_tick(
                        self._camera, landmarker)

                    # Limit FPS
                    if (time_to_sleep := self.frame_duration - time_taken) > 0:
                        sleep(time_to_sleep)

    def start_camera_loop(self) -> None:
        self._camera_thread.start()

    def stop_camera_loop(self) -> None:
        self._stop_event.set()
        self._camera_thread.join()

# TODO: Make it better
class MotionRecorder:
    def __init__(self, motion_id: int, singular: bool, hand: Literal["left", "right"], add_to_existing: bool) -> None:
        """
            Default behavior: records motion continuously after space is pressed until space is pressed again

            If singular is true: records one motion after space is pressed

            Left hand motions are singular frames!!!
        """
        self.motion_id = motion_id
        self.singular = singular
        self.hand = hand

        self._motions = []

        # Threads
        self._stop_event = Event()
        self._camera_thread = Thread(target=self._capture_loop)

    def _record_motion(self) -> None:
        scaller = StandardScaler()
        scaller.fit(self.global_deque[0])

        try:
            self._motions.append(
                list(map(scaller.transform, self.global_deque)))
        except RuntimeError:
            return

    def _save_file(self) -> None:
        np.save(
            f"training_data/{self.hand.lower()}/motion_{self.motion_id}", np.array(self._motions))

    def _show_landmarks_loop(self) -> None:
        return


    def recording_loop(self) -> None:
        import pygame
        pygame.init()

        WIDTH, HEIGHT = 1280, 720
        win = pygame.display.set_mode((WIDTH, HEIGHT))
        clock = pygame.time.Clock()

        recording = -1
        run = True
        HandLandmarker = mp.tasks.vision.HandLandmarker
        with HandLandmarker.create_from_options(self._landmarker_options) as landmarker:
            while run:
                # Frame processing
                _, frame = self._camera.read()
                mp_image = mp.Image(
                    image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

                # Detecting landmarks
                landmarker.detect_async(mp_image, int(time()*1000))

                # Pygame events
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        run = False
                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_SPACE and recording == -1:
                            recording = 20

                # Fill screen
                if recording > 0:
                    win.fill((55, 155, 55))
                else:
                    win.fill((0, 0, 0))

                # Display landmarks
                for x, y, _ in self.global_deque[-1]:
                    x_, y_ = np.array(normalized_to_px(
                        x, y, width=WIDTH, height=HEIGHT))//3
                    pygame.draw.circle(win, (255, 255, 255),
                                       (x_+WIDTH//2, y_+HEIGHT//2), 5)
                pygame.display.update()

                # Show camera
                cv2.imshow("x", frame)
                if cv2.waitKey(1) == ord("q"):
                    run = False

                # Save motion
                if self.singular:
                    if recording == 0:
                        self._record_motion()
                    if recording >= 0:
                        recording -= 1
                elif not self.single_frame_motion:
                    self._record_motion()
                else:
                    self._motions.append(
                        StandardScaler().fit_transform(self.global_deque[-1]))

                print(clock.get_fps())
                clock.tick(20)
        pygame.quit()

        self._save_file()
