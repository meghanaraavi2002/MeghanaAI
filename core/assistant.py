from brain.llm import Brain
import threading
import time
import itertools
class Vennela:
    def __init__(self):
        self.brain=Brain()
        self.thinking=False
    def animation(self):
        for dots in itertools.cycle([".","..","..."]):
            if not self.thinking:
                break
            print(f"\rVennela: Thinking{dots} ",end="",flush=True)
            time.sleep(0.5)
            print("\r"+" "*40+"\r",end="")

    def chat(self):
        print("Vennela is Ready .......")
        while True:
            user=input("\nYou:")
            if user.lower()=="exit":
                break
            self.thinking=True
            animate=threading.Thread(target=self.animation)
            animate.start()
            print("\n[Assistant] Sending Prompt To Brain...")
            answer=self.brain.think(user)
            print("[Assistant] Brain Returned Response.")
            self.thinking=True
            animate.join()
            print("\nVennela:",answer)
