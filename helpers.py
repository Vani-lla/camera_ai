import numpy as np
from sklearn.preprocessing import StandardScaler

def prepare_global_scaler() -> StandardScaler:
    X = []
    for i in range(5):
        tmp = np.load(f"training_data/{i}.npy")
        X.append(tmp)

    X = np.row_stack(X).reshape(-1, 21*2*20)

    global_scaler = StandardScaler()
    global_scaler.fit(X)

    return global_scaler

def card_to_center_coordinates(card: dict, width_offset: int = 0, height_offset: int = 0, screen_height: int = 1080) -> tuple[int, int]:
    return (
        card["TopLeftX"] + card["Width"]//2 + width_offset,
        screen_height - card["TopLeftY"] + card["Height"]//2 + height_offset
    )

def landmarks_from_index(ind: int, landmarks: list) -> list:
    if len(landmarks) == 1:
        return landmarks[0]
    else:
        return landmarks[ind]
