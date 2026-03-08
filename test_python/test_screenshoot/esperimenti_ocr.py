import cv2
import numpy as np
import pytesseract
import re
from pathlib import Path
from PIL import Image

# =========================================================
# CONFIG
# =========================================================

# Se tesseract non è nel PATH, scommenta e metti il percorso giusto:
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

IMAGE_PATH = r"test.png"
SAVE_DEBUG = True

# Crop opzionale: metti True e regola le coordinate
USE_CROP = False
CROP_X = 0
CROP_Y = 0
CROP_W = 200
CROP_H = 80


# =========================================================
# UTILS
# =========================================================

def load_image(path):
    img = Image.open(path)
    arr = np.array(img)

    if arr.ndim == 2:
        return arr

    if arr.shape[2] == 4:
        return cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)

    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def crop_roi(img):
    if not USE_CROP:
        return img.copy()

    h, w = img.shape[:2]

    x1 = max(0, CROP_X)
    y1 = max(0, CROP_Y)
    x2 = min(w, CROP_X + CROP_W)
    y2 = min(h, CROP_Y + CROP_H)

    if x1 >= x2 or y1 >= y2:
        print("Crop non valido, uso immagine intera.")
        return img.copy()

    return img[y1:y2, x1:x2].copy()


def save_debug_image(name, img):
    if not SAVE_DEBUG:
        return
    try:
        cv2.imwrite(name, img)
    except Exception as e:
        print(f"Errore salvataggio {name}: {e}")


def ensure_bgr(img):
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img


# =========================================================
# OCR PARSE
# =========================================================

def parse_ocr_number(text: str):
    if text is None:
        return None

    text = text.strip()
    if not text:
        return None

    replacements = {
        'O': '0', 'o': '0',
        'I': '1', 'l': '1', '|': '1',
        'S': '5', 's': '5',
        'B': '8',
        '€': '',
        '$': '',
        '£': '',
        ' ': '',
        '\n': '',
        '\r': '',
        ':': '',
        ';': '',
        "'": '',
    }

    for k, v in replacements.items():
        text = text.replace(k, v)

    # Cerca pattern tipo 0,02 / 0.02 / 12 / 12,5 ecc.
    candidates = re.findall(r'[-+]?\d+[.,]\d+|[-+]?\d+', text)

    if not candidates:
        return None

    # Preferisci numeri con parte decimale, poi più lunghi
    candidates.sort(key=lambda x: (("." in x or "," in x), len(x)), reverse=True)
    best = candidates[0].replace(',', '.')

    # Se per qualche motivo ci sono più punti, tieni solo l'ultimo come decimale
    if best.count('.') > 1:
        parts = best.split('.')
        best = ''.join(parts[:-1]) + '.' + parts[-1]

    try:
        return float(best)
    except ValueError:
        return None


def score_result(raw_text, parsed_value):
    score = 0

    if raw_text and raw_text.strip():
        score += 10

    if parsed_value is not None:
        score += 100

        # premia i valori piccoli decimali, tipici come 0.02
        if 0 <= parsed_value < 10:
            score += 20

        # premia se il testo grezzo contiene virgola o punto
        if ',' in raw_text or '.' in raw_text:
            score += 15

    # penalizza testo troppo sporco
    if raw_text:
        noise = len(re.sub(r'[0-9,.\s]', '', raw_text))
        score -= noise * 2

    return score


# =========================================================
# PREPROCESS
# =========================================================

def preprocess_original(img):
    return img.copy()


def preprocess_gray_thresh(img, threshold=180, scale=4):
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()

    gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    _, th = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    return th


def preprocess_adaptive(img, scale=4):
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()

    gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    th = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        2
    )
    return th


def preprocess_white_on_green(img, scale=5):
    if img.ndim == 2:
        gray = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        _, th = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
        return 255 - th

    if img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # testo bianco / quasi bianco
    mask = cv2.inRange(hsv, (0, 0, 170), (180, 80, 255))

    kernel = np.ones((2, 2), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.GaussianBlur(mask, (3, 3), 0)

    _, mask = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Tesseract spesso preferisce nero su bianco
    return 255 - mask


def preprocess_white_on_green_strong(img, scale=7):
    if img.ndim == 2:
        gray = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        _, th = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
        return 255 - th

    if img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # più permissiva per testo piccolo chiaro
    mask = cv2.inRange(hsv, (0, 0, 150), (180, 90, 255))

    kernel = np.ones((2, 2), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.medianBlur(mask, 3)

    return 255 - mask


def preprocess_green_channel(img, scale=7):
    if img.ndim == 2:
        gray = img.copy()
    else:
        if img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        gray = img[:, :, 1]  # canale verde

    gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    _, th = cv2.threshold(gray, 190, 255, cv2.THRESH_BINARY)

    return 255 - th


def preprocess_lab_light(img, scale=6):
    if img.ndim == 2:
        gray = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        _, th = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
        return 255 - th

    img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l = lab[:, :, 0]
    l = cv2.GaussianBlur(l, (3, 3), 0)
    _, th = cv2.threshold(l, 180, 255, cv2.THRESH_BINARY)

    return 255 - th


# =========================================================
# OCR
# =========================================================

def do_ocr(img, psm=7, only_numbers=True):
    whitelist = '0123456789.,' if only_numbers else ''
    config = f'--oem 3 --psm {psm}'
    if whitelist:
        config += f' -c tessedit_char_whitelist={whitelist}'
    return pytesseract.image_to_string(img, config=config)


# =========================================================
# MAIN
# =========================================================

def main():
    image_file = Path(IMAGE_PATH)
    if not image_file.exists():
        print(f"Immagine non trovata: {image_file.resolve()}")
        return

    img = load_image(str(image_file))
    img = crop_roi(img)

    save_debug_image("debug_crop.png", img)

    tests = [


        ("gray180_psm7", preprocess_gray_thresh(img, 180), 7, True),
        ("gray210_psm7", preprocess_gray_thresh(img, 210), 7, True),

    ]

    results = []

    print(f"File: {image_file.resolve()}")
    print("-" * 70)

    for name, proc, psm, only_numbers in tests:
        raw = do_ocr(proc, psm=psm, only_numbers=only_numbers)
        parsed = parse_ocr_number(raw)
        score = score_result(raw, parsed)

        results.append({
            "name": name,
            "raw": raw,
            "parsed": parsed,
            "score": score,
        })

        print(f"[{name}]")
        print(f"raw    = {raw!r}")
        print(f"parsed = {parsed}")
        print(f"score  = {score}")
        print()

        save_debug_image(f"debug_{name}.png", proc)

    # ordina per score migliore
    results.sort(key=lambda x: x["score"], reverse=True)

    print("=" * 70)
    print("MIGLIORI RISULTATI")
    print("=" * 70)

    for r in results[:5]:
        print(f"{r['name']}: parsed={r['parsed']} score={r['score']} raw={r['raw']!r}")

    best = results[0]
    print("\n" + "=" * 70)
    print("RISULTATO FINALE")
    print("=" * 70)
    print(f"Metodo : {best['name']}")
    print(f"Raw    : {best['raw']!r}")
    print(f"Parsed : {best['parsed']}")
    print(f"Score  : {best['score']}")

    print("\nFine test.")


if __name__ == "__main__":
    main()