"""Operations to help identify items on screen"""

import io
import logging
import os
import sys
from time import sleep
import subprocess
import numpy as np
import cv2
import time
import cv2
import numpy as np
from PIL import Image, ImageGrab
#from tesserocr import PyTessBaseAPI, PSM, OEM
import pytesseract
from PIL import Image
import os
import re
from typing import Optional, Union
from collections import Counter

Number = Union[int, float]

_OCR_MAP = str.maketrans({
    "O": "0", "o": "0",
    "I": "1", "l": "1", "|": "1", "¡": "1",
    "S": "5", "s": "5",
    "B": "8",
    "Z": "2",
    "D": "0",  # a volte OCR scambia 0 con D
})

def parse_ocr_number(text: str, *, allow_negative: bool = True) -> Optional[Number]:
    if not text:
        return None

    t = text.strip().translate(_OCR_MAP)

    # Tieni solo caratteri utili
    keep = re.sub(r"[^0-9\-\+\.,\s€$£]", " ", t)

    # Token numerici SENZA attraversare spazi
    candidates = re.findall(r"[-+]?(?:\d+[.,]\d+|\d+)", keep)

    if not candidates:
        return None

    if not allow_negative:
        candidates = [c.lstrip("+-") for c in candidates]

    # Preferisci:
    # 1) numeri con parte decimale
    # 2) più cifre
    def score(c: str):
        has_decimal = 1 if ("," in c or "." in c) else 0
        digits = len(re.sub(r"\D", "", c))
        return (has_decimal, digits)

    c = max(candidates, key=score)

    # Normalizza separatori
    if "." in c and "," in c:
        last_dot = c.rfind(".")
        last_com = c.rfind(",")
        dec = "." if last_dot > last_com else ","
        thou = "," if dec == "." else "."
        c = c.replace(thou, "")
        c = c.replace(dec, ".")
    elif "," in c:
        c = c.replace(",", ".")
    # se c'è solo '.' lo lasciamo così

    c = re.sub(r"[^0-9\-\+\.]", "", c)

    if c.count(".") > 1:
        parts = c.split(".")
        c = "".join(parts[:-1]) + "." + parts[-1]

    try:
        v = float(c)
    except ValueError:
        return None

    if v.is_integer():
        return int(v)
    return v


# Path default (se non hai messo Tesseract nel PATH di sistema)
_TESS_PATH = r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"
if os.path.exists(_TESS_PATH):
    pytesseract.pytesseract.tesseract_cmd = _TESS_PATH

from poker.tools.helper import memory_cache, get_dir
from poker.tools import constants as const
from poker.tools.mongo_manager import MongoManager
from poker.tools.vbox_manager import VirtualBoxController

log = logging.getLogger(__name__)
is_debug = False  # used for saving images for debug purposes

if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
    tesserpath = os.path.join(get_dir('codebase'), 'tessdata')
else:
    tesserpath = os.path.join(get_dir('codebase'), '..', 'tessdata')

#api = PyTessBaseAPI(path=tesserpath,
#                    psm=PSM.SINGLE_LINE,
#                    oem=OEM.LSTM_ONLY)
def ocr_text(pil_img: Image.Image, psm: int = 6) -> str:
    # testo generico
    return pytesseract.image_to_string(pil_img, config=f"--oem 3 --psm {psm}")

def ocr_digits(pil_img: Image.Image, psm: int = 7) -> str:
    # numeri (stack, pot, ecc.)
    return pytesseract.image_to_string(
        pil_img,
        config=f"--oem 3 --psm {psm} -c tessedit_char_whitelist=0123456789.,"
    )

def find_template_on_screen(template, screenshot, threshold, extended=False):
    """Find template on screen"""
    res = cv2.matchTemplate(screenshot, template, cv2.TM_SQDIFF_NORMED)
    loc = np.where(res <= threshold)
    min_val, _, min_loc, _ = cv2.minMaxLoc(res)

    bestFit = min_loc
    count = 0
    points = []
    for pt in zip(*loc[::-1]):
        # cv2.rectangle(img, pt, (pt[0] + w, pt[1] + h), (0,0,255), 2)
        count += 1
        points.append(pt)
    return count, points, bestFit, min_val


@memory_cache
def load_table_template_cached(table_name):
    """Load template from mongodb as cv2 image"""
    mongo = MongoManager()
    table = mongo.get_table(table_name=table_name)
    return table


def get_table_template_image(table_name='default', label='topleft_corner'):
    """Load template from mongodb as cv2 image"""
    mongo = MongoManager()
    table = mongo.get_table(table_name=table_name)
    template_pil = Image.open(io.BytesIO(table[label]))
    template_cv2 = cv2.cvtColor(np.array(template_pil), cv2.COLOR_BGR2RGB)
    return template_cv2


def get_ocr_float(img_orig, fast=False, thresholds=[76, 100,  180, 190,200, 210]):
    """Return float value from image. -1.0f when OCR failed"""
    return get_ocr_number(img_orig, fast, thresholds)


def prepareImage(img_orig, binarize=True, threshold=76):
    """Prepare image for OCR"""

    def binarize_array_opencv(image, threshold):
        """Binarize image from gray channel with 76 as threshold"""
        img = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        _, thresh2 = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY_INV)
        return Image.fromarray(thresh2)

    basewidth = 300
    wpercent = (basewidth / float(img_orig.size[0]))
    hsize = int((float(img_orig.size[1]) * float(wpercent)))
    img_resized = img_orig.convert('L').resize(
        (basewidth, hsize), Image.LANCZOS)
    if binarize:
        img_resized = binarize_array_opencv(img_resized, threshold)

   
    if is_debug:
        pics_path = "log/pics"
        try:
            if not os.path.exists(pics_path):
                os.makedirs(pics_path)
        except OSError:
            log.error("Creation of the directory %s failed" % pics_path)
            sys.exit(1)

        img_orig.save('log/pics/img_orig.png')
        img_resized.save('log/pics/img_resized.png')

        log.debug("ocr images prepared")

    return img_resized


def _to_pil(img):
    """Converte numpy array (OpenCV) in PIL Image se necessario."""
    if isinstance(img, Image.Image):
        return img

    if isinstance(img, np.ndarray):
        if len(img.shape) == 2:  # grayscale
            return Image.fromarray(img)
        else:
            # BGR (opencv) -> RGB
            return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    raise TypeError("Unsupported image type for OCR")


def get_ocr_number2(img_orig, fast=False):
    """OCR numerico usando pytesseract (replacement di tesserocr)."""
    
    pil_img = _to_pil(img_orig)

    whitelist = "0123456789.$£B"

    result = pytesseract.image_to_string(
        pil_img,
        config=f"--oem 3 --psm 7 -c tessedit_char_whitelist={whitelist}"
    )

    result = pytesseract.image_to_string(
        pil_img
    )

    return result.strip()

def choose_best_number(lst):

    nums = [x for x in lst if x is not None]

    if not nums:
        return None

    # 1️⃣ se uno compare più volte → prendilo
    counter = Counter(nums)
    value, count = counter.most_common(1)[0]

    if count > 1:
        return value

    # 2️⃣ tutti diversi → trova quello più vicino agli altri
    best = None
    best_score = float("inf")

    for x in nums:
        score = sum(abs(x - y) for y in nums)
        if score < best_score:
            best_score = score
            best = x

    return best



def get_ocr_number(img_orig, fast=False, thresholds = [76, 100,  180, 190,200, 210]):
    """Return float value from image. -1.0f when OCR failed"""




    lst = []

    for th in thresholds:

        img_resized = prepareImage(img_orig, binarize=True, threshold=th)

        read_o = get_ocr_number2(img_resized)
        read = parse_ocr_number(read_o)

        lst.append(read)

    value = choose_best_number(lst)

    try:
        if value is not None:
            return float(value)
        raise Exception("Errore conv numero")
    except :
        return -1


        if fast:
            return -1
        # , img_min, img_mod, img_med, img_sharp]
        images = [img_orig, img_resized]
        i = 0
        while i < 2:
            j = 0
            while j < len(images):
                lst.append(
                    get_ocr_number2(images[j]).
                    strip().replace('$', '').replace('£', '').replace('€', '').replace('B', '').replace('\n', '').replace(':', ''))
                j += 1
            i += 1

    log.debug(lst)
    for element in lst:
        try:
            if element is not None:
                num = float(element)
        except :
            pass
            # log.warning(f"Not recognized: {element}")
    return -1.0

def fast_screenshot(adb_path="adb"):
    result = subprocess.run(
        [adb_path, "exec-out", "screencap", "-p"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )

    if result.returncode != 0:
        raise RuntimeError(f"ADB error: {result.stderr.decode(errors='ignore')}")

    if not result.stdout:
        raise RuntimeError("ADB returned empty screenshot")

    # Fix newline Windows
    data = result.stdout.replace(b"\r\r\n", b"\n")

    img_bgr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)

    if img_bgr is None:
        raise RuntimeError("Failed to decode screenshot")

    # Convert BGR -> RGB
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # Convert to PIL Image (IMPORTANTE per ImageQt)
    pil_image = Image.fromarray(img_rgb)

    return pil_image


def take_screenshot(virtual_box=False):
    """
    Take screenshot directly from screen or via virtualbox

    Args:
        virtual_box: bool

    Returns:
        PIL screenshot

    """
    if not virtual_box:
        #log.debug("Calling screen grabber")
        #screenshot = ImageGrab.grab()
        #log.debug("Direct screenshot successful")
        try:
            screenshot = fast_screenshot()
        except Exception as exc:
            log.warning(f"Fast screenshot failed: {exc}. Falling back to ImageGrab.")
            screenshot = None

        if screenshot is None:
            print("Errore screenshot")
            

    else:  # virtual_box
        try:
            vb = VirtualBoxController()
            screenshot = vb.get_screenshot_vbox()
            log.debug("Screenshot taken from virtual machine")
        except:
            #log.warning(
              #  "No virtual machine found. Press SETUP to re initialize the VM controller")
            # gui_signals.signal_open_setup.emit(p,L)
            screenshot = ImageGrab.grab()
    return screenshot

def normalize_rect(x1, y1, x2, y2):
    x1_ = min(x1, x2)
    x2_ = max(x1, x2)
    
    y1_ = min(y1,y2)
    y2_ = max(y1,y2)

    return x1_, y1_, x2_, y2_

def check_cropping(screenshot_list, top_left_corner_img):
    """Checks if screenshots are cropped and match the template 'icon'"""
    try:
        log.info("Checking cropping for '" + str(len(screenshot_list)) + "' images.")
        
        if len(screenshot_list) == 0: return False
        if top_left_corner_img.size == 0: return False

        any_too_big = any((s.width > const.CROP_WIDTH and s.height > const.CROP_HEIGHT) for s in screenshot_list)
        if any_too_big: return False
        
        for screenshot in screenshot_list:
            img = cv2.cvtColor(np.array(screenshot), cv2.COLOR_BGR2RGB)
            count, _, _, _ = find_template_on_screen(top_left_corner_img, img, 0.01)
            if count != 1: return False
    except Exception as e:
        log.exception(e)
        return False
    finally:
        log.info("Done.")

    return True

def crop_screenshot_with_topleft_corner(original_screenshot, topleft_corner, useSleep = True):

    #con il cellulare prendiamo tutto senza crop
    cropped_screenshot = original_screenshot
    return cropped_screenshot , (0, 0)

    log.debug("Cropping top left corner")
    img = cv2.cvtColor(np.array(original_screenshot), cv2.COLOR_BGR2RGB)
    count, points, _, _ = find_template_on_screen(topleft_corner, img, 0.01)

    if count == 1:
        tlc = points[0]
        log.debug(f"Found to left corner at {tlc}")
        cropped_screenshot = original_screenshot.crop(
            (tlc[0], tlc[1], tlc[0] + const.CROP_WIDTH, tlc[1] + const.CROP_HEIGHT))
        return cropped_screenshot, tlc
    elif count > 1:
        log.warning(
            "Multiple top left corners found. That doesn't work unfortunately at this point. Make sure only one table is visible.")
        return None, None
    else:
        log.warning("No top left corner found")
        if useSleep: sleep(5)
        return None, None


def binary_pil_to_cv2(img,name="nintest"):
    cv2_img =cv2.cvtColor(np.array(Image.open(io.BytesIO(img))), cv2.COLOR_BGR2RGB)
    cv2.imwrite("img_export/"+name+'.png', cv2_img) 

    return cv2.cvtColor(np.array(Image.open(io.BytesIO(img))), cv2.COLOR_BGR2RGB)


def pil_to_cv2(img):
    return cv2.cvtColor(np.array(img), cv2.COLOR_BGR2RGB)


def cv2_to_pil(img):
    return Image.fromarray(img)


def rotate_image(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(
        image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result


def check_if_image_in_range(img, screenshot, x1, y1, x2, y2, extended=False, threshold=0.01):
    cropped_screenshot = screenshot.crop((x1, y1, x2, y2))
    cropped_screenshot = pil_to_cv2(cropped_screenshot)
    count, _, _, _ = find_template_on_screen(
        img, cropped_screenshot, threshold, extended=extended)
    return count >= 1

def is_template_in_search_area_scaled(table_dict, screenshot, image_name, image_area, player=None, extended=False):

    template_cv2 = binary_pil_to_cv2(table_dict[image_name], name=image_name)

    if player:
        try:
            search_area = table_dict[image_area][player]
        except KeyError as exc:
            raise KeyError(
                f"The table mapping is missing data for player {player} and {image_area}."
            ) from exc
    else:
        search_area = table_dict[image_area]

    scales = [1.2]

    for scale in scales:

        new_w = int(template_cv2.shape[1] * scale)
        new_h = int(template_cv2.shape[0] * scale)

        if new_w < 2 or new_h < 2:
            continue

        template_scaled = cv2.resize(
            template_cv2,
            (new_w, new_h),
            interpolation=cv2.INTER_AREA
        )

        try:
            is_in_range = check_if_image_in_range(
                template_scaled,
                screenshot,
                search_area['x1'],
                search_area['y1'],
                search_area['x2'],
                search_area['y2'],
                extended=extended
            )



            if is_in_range:
                print(image_name, "found at scale", scale)
                return True

        except Exception:
            pass

    return False

def is_template_in_search_area(table_dict, screenshot, image_name, image_area, player=None, extended=False,threshold=0.01):
    template_cv2 = binary_pil_to_cv2(table_dict[image_name], name=image_name)
    if player:
        try:
            search_area = table_dict[image_area][player]
        except KeyError as exc:
            raise KeyError(f"The table mapping is missing data for player {player} and {image_area}."
                           "Please fix the table mapping.") from exc
    else:
        search_area = table_dict[image_area]
    try:
        is_in_range = check_if_image_in_range(template_cv2, screenshot,
                                              search_area['x1'], search_area['y1'], search_area['x2'], search_area['y2'],
                                              extended=extended, threshold=threshold)
    except Exception as exc:
        x = search_area['x2'] - search_area['x1']
        y = search_area['y2'] - search_area['y1']
        xt = template_cv2.shape[1]
        yt = template_cv2.shape[0]
        if x < xt or y < yt:
            raise RuntimeError(f"Search area for {image_name} {player} is too small. It is {x}x{y} but the template is {xt}x{yt}."
                               ) from exc
        raise RuntimeError(f"The table has an missing template for {image_name}."
                           ) from exc

    return is_in_range


def ocr(screenshot, image_area, table_dict, player=None, fast=False, thresholds=[76, 100,  180, 190,200, 210]):
    """
    get ocr of area of screenshot

    Args:
        screenshot: pil image
        image_area: area name
        table_dict: table dict
        player: player number started from 0

    Returns:
        float

    """
    if player:
        try:
            search_area = table_dict[image_area][player]
        except KeyError:
            log.error(f"Missing table entry for {image_area} {player}. "
                      f"Please select it from the screenshot and press the corresponding button to add it to the "
                      f"table template. ")
            return 0
    else:
        search_area = table_dict[image_area]
    cropped_screenshot = screenshot.crop(
        (search_area['x1'], search_area['y1'], search_area['x2'], search_area['y2']))
    return get_ocr_float(cropped_screenshot, fast, thresholds)
