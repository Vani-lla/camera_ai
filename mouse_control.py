from collections import deque
from time import sleep
import numpy as np
import pyautogui
from helpers import CONSTANTS
from motions import HandMotions

class MouseController:
    def __init__(self, screen_settings: dict[str, int]) -> None:
        self.screen_settings = screen_settings

        self.status = (0, 0)

        self.hand_constant = .25
        self.center = .5

        self.card_constraints = (0, 0)

    def _get_current_selected_card(self, num_of_cards: int, x: float) -> int:
        x_ = abs(x - self.center + self.hand_constant)

        distance = 2*self.hand_constant/num_of_cards

        return (x_//distance) % num_of_cards

    def calibrate(self, hand: deque[tuple[float, float, float]]) -> None:
        hand_x = hand[:, 0]

        self.center = np.average(hand_x)
        self.hand_constant = np.max(hand_x) - np.min(hand_x)

    def update_card_constraints(self, cards: list[dict]) -> tuple[int, int]:
        def get_x(card: dict) -> int:
            return card["TopLeftX"]

    def update_hand_constant(self, hand: np.ndarray[np.ndarray]) -> float:
        self.hand_constant = np.min(hand[:, 0])
        return self.hand_constant

    def update_center(self, hand: np.ndarray[np.ndarray]) -> float:
        self.center = np.average(hand[:, 0])
        return self.center

    def next_turn(self):
        pyautogui.click(CONSTANTS["skip"], _pause=False)

    def choosing(self, x: float):
        screen_center = self.screen_settings["width"]//2

        distance = self.center - x

        # pyautogui.moveTo(, self.screen_settings["height"])

    def mouse_coordinate(self):
        return pyautogui.position()


if __name__ == "__main__":
    import json

    with open("settings/settings.json", "r") as settings:
        controller = MouseController(json.load(settings)["screen"])

    while True:
        print(controller.mouse_coordinate())
        controller.choosing()
        sleep(2)
