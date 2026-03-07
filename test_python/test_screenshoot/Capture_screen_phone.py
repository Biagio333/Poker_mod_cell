import subprocess
import numpy as np
import cv2
import time

def fast_screenshot():
    result = subprocess.run(
        ["adb", "exec-out", "screencap", "-p"],
        stdout=subprocess.PIPE
    )

    # Fix newline Windows
    data = result.stdout.replace(b"\r\r\n", b"\n")

    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    return img

while True:
    start = time.time()

    img = fast_screenshot()

    if img is None:
        print("Errore screenshot")
        continue

    cv2.imshow("Phone Live", img)

    # Premi Q per uscire
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Stampa FPS
    elapsed = time.time() - start
    fps = 1 / elapsed
    print(f"FPS: {fps:.2f}")

cv2.destroyAllWindows()