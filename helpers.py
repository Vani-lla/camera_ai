from json import load
from typing import Literal
import numpy as np
from sklearn.preprocessing import StandardScaler


with open("settings/settings.json") as file:
    SETTINGS: dict = load(file)
CONSTANTS = {
    "skip": (1668, 537),
}


def prepare_global_scaler() -> StandardScaler:
    X = []
    for i in range(-1, 3):
        tmp = np.load(f"training_data/right/motion_{i}.npy")
        X.append(tmp)

    X = np.row_stack(X).reshape(-1, 21*3*20)

    global_scaler = StandardScaler()
    global_scaler.fit(X)

    return global_scaler


def get_data(hand: Literal["left", "right"], num: int) -> tuple[np.ndarray, np.ndarray]:
    X, Y = [], []
    for i in range(-1, num):
        tmp = np.load(f"training_data/{hand}/motion_{i}.npy")
        X.append(tmp)
        for _ in range(len(tmp)):
            Y.append(i)

    if hand == "right":
        X = np.row_stack(X).reshape(-1, 21*3*20)
    else:
        X = np.row_stack(X).reshape(-1, 21*3)
    Y = np.array(Y)

    return X, Y


def card_to_center_coordinates(card: dict, width_offset: int = 0, height_offset: int = 0, screen_height: int = 1080) -> tuple[int, int]:
    return (
        card["TopLeftX"] + card["Width"]//2 + width_offset,
        screen_height - card["TopLeftY"] + card["Height"]//2 + height_offset
    )

def normalized_to_px(x: float, y: float, width: int = SETTINGS["screen"]["width"], height: int = SETTINGS["screen"]["height"]) -> tuple[int, int]:
    return int((1-x)*width), int(y*height)


if __name__ == "__main__":
    print(get_data("left", 2)[0].shape)
