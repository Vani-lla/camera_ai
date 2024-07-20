from threading import Thread

from requests import request
from time import sleep
from os import system

from helpers import card_to_center_coordinates

# 4, 4 ---- replacing - position to be sure
# 260 topY --- placed ally
# below 260 --- in hand
# 450 --- atacking/defending

class Game:
    def __init__(self, port: int, interval: float) -> None:
        self.PORT = port
        self.interval = interval

        self.card_json: list[dict] = []

        self._thread = Thread(target=self._updater)

    def _updater(self) -> None:
        """
            Only to use inside of a Thread
            
            Sends requests to LOR with a given interval
        """
        while True:
            try:
                response = request(
                    "GET", f"http://127.0.0.1:{self.PORT}/positional-rectangles")
                self.card_json = response.json()["Rectangles"]
            except Exception as e:
                print(f"Error in response: {e}")

    def start(self) -> None:
        self._thread.start()

    @property
    def local_player_hand(self) -> list[dict]:
        return list(map(card_to_center_coordinates, filter(lambda c: c["TopLeftY"] < 260 and c["LocalPlayer"] and c["CardCode"] != "face", self.card_json)))

    @property
    def local_player_bench(self) -> list[tuple[int, int]]:
        return list(map(card_to_center_coordinates, filter(lambda c: c["TopLeftY"] == 260 and c["LocalPlayer"] and c["CardCode"] != "face", self.card_json)))

    @property
    def local_player_select_cards(self) -> list[tuple[int, int]]:
        return list(map(card_to_center_coordinates, filter(lambda c: c["TopLeftY"] > 260 and c["LocalPlayer"] and c["CardCode"] != "face", self.card_json)))

    @property
    def enemy_player_hand(self) -> list[dict]:
        ...

    @property
    def enemy_player_bench(self) -> list[dict]:
        ...

game = Game(21337, 1.0)
game.start()

while True:
    system("cls")

    print(game.local_player_hand)
    print(game.local_player_bench)
    print(game.local_player_select_cards)

    print(set(c["TopLeftY"] for c in game.card_json if c["LocalPlayer"]))

    sleep(1)