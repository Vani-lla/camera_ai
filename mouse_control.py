from collections import deque
from time import sleep
import numpy as np
import pyautogui
from motions import HandMotions

CONSTANTS = {
    "skip": (1668, 537),
}


class MouseController:
    def __init__(self, screen_settings: dict[str, int]) -> None:
        self.screen_settings = screen_settings

        self.status = (0, 0)

        self.hands_constant = .25
        self.center = .5

        self.card_constraints = (0, 0)

    def update_card_constraints(self, cards: list[dict]) -> tuple[int, int]:
        def get_x(card: dict) -> int:
            return card["TopLeftX"]
        
        # left_constraint = min(cards, key=get_x)
        # right_constraint_index = 
        # right_constraint = np.argmin(list(map(get_x, cards)))

    def update_hand_constant(self, hand: np.ndarray[np.ndarray]) -> float:
        self.hands_constant = np.min(hand[:, 0])
        return self.hands_constant

    def update_center(self, hand: np.ndarray[np.ndarray]) -> float:
        self.center = np.average(hand[:, 0])
        return self.center

    def next_turn(self):
        pyautogui.click(*CONSTANTS["skip"], _pause=False)

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
