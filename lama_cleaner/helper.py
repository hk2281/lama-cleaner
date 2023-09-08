import io
import os
import sys
from typing import List, Optional

from urllib.parse import urlparse
import cv2
import math
from PIL import Image, ImageOps, PngImagePlugin, ImageChops
import numpy as np
import torch
from lama_cleaner.const import MPS_SUPPORT_MODELS
from loguru import logger
from torch.hub import download_url_to_file, get_dir
import hashlib


def find_intersection(p1, p2, q1, q2):
    # p1 и p2 - координаты первого отрезка (p1 = (x1, y1), p2 = (x2, y2))
    # q1 и q2 - координаты второго отрезка (q1 = (x3, y3), q2 = (x4, y4))

    # Вычисляем векторы направления отрезков
    vector1 = (p2[0] - p1[0], p2[1] - p1[1])
    vector2 = (q2[0] - q1[0], q2[1] - q1[1])

    # Вычисляем определитель матрицы из векторов направления
    determinant = vector1[0] * vector2[1] - vector1[1] * vector2[0]

    # Если определитель равен нулю, отрезки параллельны и не пересекаются
    if determinant == 0:
        return None

    # Вычисляем параметры t и u для точки пересечения
    t = ((q1[0] - p1[0]) * vector2[1] - (q1[1] - p1[1]) * vector2[0]) / determinant
    u = ((q1[0] - p1[0]) * vector1[1] - (q1[1] - p1[1]) * vector1[0]) / determinant

    # Если 0 <= t <= 1 и 0 <= u <= 1, то отрезки пересекаются
    if 0 <= t <= 1 and 0 <= u <= 1:
        intersection_x = p1[0] + t * vector1[0]
        intersection_y = p1[1] + t * vector1[1]
        return (intersection_x, intersection_y)
    else:
        return None


def mask_line_extender(p1,p2,target_X, target_Y):
  x1, y1 = p1
  x2, y2 = p2
  # X = 5 + (25 - 15) * (15 - 5) / (5 - 15)
  X = x1 + (target_Y - y1) * (x2 - x1) / (y2 - y1)
  X = 0 if X < 0 else X
  # Y = 15 + (25 - 5) * (5 - 15) / (15 - 5)
  Y = y1 + (target_X - x1) * (y2 - y1) / (x2 - x1)
  Y = 0  if  Y < 0 else Y

  print('ex',X,Y)
  return (X,Y)

def find_intersection_angle(p1,p2,q1,q2):
    # Вычисляем векторы направления для отрезков
    vector1 = (p2[0] - p1[0], p2[1] - p1[1])
    vector2 = (q2[0] - q1[0], q2[1] - q1[1])

    # Вычисляем скалярное произведение векторов
    dot_product = vector1[0] * vector2[0] + vector1[1] * vector2[1]

    # Вычисляем длины векторов
    magnitude1 = math.sqrt(vector1[0] ** 2 + vector1[1] ** 2)
    magnitude2 = math.sqrt(vector2[0] ** 2 + vector2[1] ** 2)

    # Вычисляем угол между отрезками в радианах
    angle_rad = math.acos(dot_product / (magnitude1 * magnitude2))

    # Преобразуем угол в градусы
    angle_deg = math.degrees(angle_rad)

    print(f"Угол между отрезками: {90-angle_deg} градусов")
    return 90-angle_deg


def get_mask(img_path:np.array=None, model=None) -> None:

  if img_path is None:
    raise Exception('img path is empty set corrent img_path: arg')
  if model is None:
    raise Exception('passed model not fit here, set corrent model: arg')

  pred_result = model(img_path)
  yolo_contours = []
  for r in pred_result:
    yolo_img = r.orig_img
    yolo_mask = r.masks.data
    yolo_mask = yolo_mask.cpu().numpy().astype('uint8')
    yolo_contour, _ = cv2.findContours(yolo_mask[0], cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    yolo_contours.append(yolo_contour[0])
  result = np.zeros_like(yolo_img)
  image_with_overlaid_predictions = yolo_img.copy()

  for contour in yolo_contours:

      print(type(contour),contour.shape)

      direction = "vertical"  # или "horizontal" для горизонтального направления

      x_min = contour[:, :, 0].min()
      x_max = contour[:, :, 0].max()
      y_min = contour[:, :, 1].min()
      y_max = contour[:, :, 1].max()
      mid_x = (x_min + x_max) // 2
      mid_y = (y_min + y_max) // 2
      cv2.line(image_with_overlaid_predictions, (mid_x-200, mid_y), (mid_x+200,mid_y), (255, 0, 0), 2)
      print(x_min,y_min,x_max,y_max)
      intersections = []

      x1, y1 = mid_x-200, mid_y
      x2, y2 = mid_x+200, mid_y

      m = (y2 - y1) / (x2 - x1)
      b = y1 - m * x1
      print(m,b)
      intersections = []

      for i in range(len(contour) - 1):
        try:
          x1, y1 = contour[i][0]
          x2, y2 = contour[i + 1][0]

        except Exception as e:
          print(e)
        # print(x1,y1,x2,y2)

        my_intersaction_points = find_intersection((mid_x-200, mid_y),(mid_x+200, mid_y),(x1,y1),(x2,y2))
        if my_intersaction_points:
          intersections.append(my_intersaction_points)

      if len(intersections) > 1:
        intersections = intersections[::len(intersections)-1]
      for point in intersections:
          print(f"Точка пересечения: ({point[0]}, {point[1]})")
      intersection_angle = find_intersection_angle((mid_x-200, mid_y),(mid_x+200, mid_y),(x_min,y_max),(x_max,y_min))


      # находим длинну проекции
      # Длина проекции = |AB| * cos(θ)
      # находим длинну отрезка
      A = intersections[0]
      B = intersections[1]
      line_len = B[0]-A[0]
      print('len',line_len)
      # находим длинну проекции
      projection_len = int(abs(abs(line_len)*math.cos(intersection_angle)))
      print('roj. len', projection_len)
      if projection_len <= 0:
        projection_len=15
      print("y_min",x_min)
      cv2.line(result, (int(0),480), (0,480), (255, 150, 255), thickness=5)
      extendet_X,extendet_Y = mask_line_extender((x_min,y_max),(x_max,y_min),yolo_img.shape[1],yolo_img.shape[0])
      cv2.line(result, (x_min,y_max), (x_max,y_min), (255, 255, 255), thickness=projection_len+5)
      cv2.line(result, (int(extendet_X),yolo_img.shape[0]), (yolo_img.shape[1],int(extendet_Y)), (255, 255, 255), thickness=projection_len+5)
      cv2.drawContours(result, [contour],-1, (255,255,255),thickness=cv2.FILLED)
      print('mask type', type(result))
      return result


def add_watermark(target_image_path=None, watermark_image_path=None, output_image_path=None, opacity=28):

    # Открываем целевое и водяное изображения
    target_image = Image.fromarray(np.uint8(target_image_path))
    # print(watermark_image_path)
    # target_image = Image.open(target_image_path)
    watermark_image_path = watermark_image_path[..., ::-1]
    watermark_image = Image.fromarray(np.uint8(watermark_image_path))

    # Определяем размеры целевого изображения
    target_width, target_height = target_image.size

    # Определяем соотношение масштабирования для водяного знака
    scale_factor = min(target_width / watermark_image.width, target_height / watermark_image.height)

    # Масштабируем водяной знак
    new_width = int(watermark_image.width * scale_factor)
    new_height = int(watermark_image.height * scale_factor)
    watermark_image = watermark_image.resize((new_width, new_height), Image.Resampling.LANCZOS)

    # Рассчитываем прозрачность водяного знака
    watermark_opacity = int((opacity /100) * 255)
    watermark_image.putalpha(watermark_opacity)

    # Создаем копию целевого изображения для редактирования
    output_image = target_image.copy()

    # Рассчитываем координаты для размещения водяного знака по центру
    # x = (target_width - watermark_image.width) // 2
    # y = (target_height - watermark_image.height) // 2
    top_x = (target_width - watermark_image.width) // 2
    top_y = target_height // 4 - watermark_image.height // 2

    # Рассчитываем координаты для размещения среднего водяного знака
    middle_x = (target_width - watermark_image.width) // 2
    middle_y = (target_height - watermark_image.height) // 2

    # Рассчитываем координаты для размещения нижнего водяного знака
    bottom_x = (target_width - watermark_image.width) // 2
    bottom_y = (3 * target_height) // 4 - watermark_image.height // 2
    # watermark_image = watermark_image.rotate(45, expand=True)
    # Наносим водяной знак на изображение
    output_image.paste(watermark_image,(top_x,top_y),watermark_image)
    output_image.paste(watermark_image,(middle_x,middle_y),watermark_image)
    output_image.paste(watermark_image,(bottom_x,bottom_y),watermark_image)
    # Сохраняем результат

    # output_image.save(output_image_path)
    
    return(np.array(output_image))


def trim(im):
    im = Image.fromarray(np.uint8(im))
    bg = Image.new(im.mode, im.size, im.getpixel((0,0)))
    diff = ImageChops.difference(im, bg)
    diff = ImageChops.add(diff, diff, 2.0, -100)
    bbox = diff.getbbox()
    if bbox:
        return np.array(im.crop(bbox))
    else:
        # Failed to find the borders, convert to "RGB"
        return np.array(trim(im.convert('RGB')))


def md5sum(filename):
    md5 = hashlib.md5()
    with open(filename, "rb") as f:
        for chunk in iter(lambda: f.read(128 * md5.block_size), b""):
            md5.update(chunk)
    return md5.hexdigest()


def switch_mps_device(model_name, device):
    if model_name not in MPS_SUPPORT_MODELS and str(device) == "mps":
        logger.info(f"{model_name} not support mps, switch to cpu")
        return torch.device("cpu")
    return device


def get_cache_path_by_url(url):
    parts = urlparse(url)
    hub_dir = get_dir()
    model_dir = os.path.join(hub_dir, "checkpoints")
    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)
    filename = os.path.basename(parts.path)
    cached_file = os.path.join(model_dir, filename)
    return cached_file


def download_model(url, model_md5: str = None):
    cached_file = get_cache_path_by_url(url)
    if not os.path.exists(cached_file):
        sys.stderr.write('Downloading: "{}" to {}\n'.format(url, cached_file))
        hash_prefix = None
        download_url_to_file(url, cached_file, hash_prefix, progress=True)
        if model_md5:
            _md5 = md5sum(cached_file)
            if model_md5 == _md5:
                logger.info(f"Download model success, md5: {_md5}")
            else:
                try:
                    os.remove(cached_file)
                    logger.error(
                        f"Model md5: {_md5}, expected md5: {model_md5}, wrong model deleted. Please restart lama-cleaner."
                        f"If you still have errors, please try download model manually first https://lama-cleaner-docs.vercel.app/install/download_model_manually.\n"
                    )
                except:
                    logger.error(
                        f"Model md5: {_md5}, expected md5: {model_md5}, please delete {cached_file} and restart lama-cleaner."
                    )
                exit(-1)

    return cached_file


def ceil_modulo(x, mod):
    if x % mod == 0:
        return x
    return (x // mod + 1) * mod


def handle_error(model_path, model_md5, e):
    _md5 = md5sum(model_path)
    if _md5 != model_md5:
        try:
            os.remove(model_path)
            logger.error(
                f"Model md5: {_md5}, expected md5: {model_md5}, wrong model deleted. Please restart lama-cleaner."
                f"If you still have errors, please try download model manually first https://lama-cleaner-docs.vercel.app/install/download_model_manually.\n"
            )
        except:
            logger.error(
                f"Model md5: {_md5}, expected md5: {model_md5}, please delete {model_path} and restart lama-cleaner."
            )
    else:
        logger.error(
            f"Failed to load model {model_path},"
            f"please submit an issue at https://github.com/Sanster/lama-cleaner/issues and include a screenshot of the error:\n{e}"
        )
    exit(-1)


def load_jit_model(url_or_path, device, model_md5: str):
    if os.path.exists(url_or_path):
        model_path = url_or_path
    else:
        model_path = download_model(url_or_path, model_md5)

    logger.info(f"Loading model from: {model_path}")
    try:
        model = torch.jit.load(model_path, map_location="cpu").to(device)
    except Exception as e:
        handle_error(model_path, model_md5, e)
    model.eval()
    return model


def load_model(model: torch.nn.Module, url_or_path, device, model_md5):
    if os.path.exists(url_or_path):
        model_path = url_or_path
    else:
        model_path = download_model(url_or_path, model_md5)

    try:
        logger.info(f"Loading model from: {model_path}")
        state_dict = torch.load(model_path, map_location="cpu")
        model.load_state_dict(state_dict, strict=True)
        model.to(device)
    except Exception as e:
        handle_error(model_path, model_md5, e)
    model.eval()
    return model


def numpy_to_bytes(image_numpy: np.ndarray, ext: str) -> bytes:
    data = cv2.imencode(
        f".{ext}",
        image_numpy,
        [int(cv2.IMWRITE_JPEG_QUALITY), 100, int(cv2.IMWRITE_PNG_COMPRESSION), 255],
    )[1]
    image_bytes = data.tobytes()
    return image_bytes


def pil_to_bytes(pil_img, ext: str, quality: int = 95, exif_infos={}) -> bytes:
    with io.BytesIO() as output:
        kwargs = {k: v for k, v in exif_infos.items() if v is not None}
        if ext == "png" and "parameters" in kwargs:
            pnginfo_data = PngImagePlugin.PngInfo()
            pnginfo_data.add_text("parameters", kwargs["parameters"])
            kwargs["pnginfo"] = pnginfo_data

        pil_img.save(
            output,
            format=ext,
            quality=quality,
            **kwargs,
        )
        image_bytes = output.getvalue()
    return image_bytes


def load_img(img_bytes, gray: bool = False, return_exif: bool = False):
    alpha_channel = None
    
    if type(img_bytes) == np.ndarray:
        image = Image.fromarray(np.uint8(img_bytes))
    else:
        image = Image.open(io.BytesIO(img_bytes))

    if return_exif:
        info = image.info or {}
        exif_infos = {"exif": image.getexif(), "parameters": info.get("parameters")}

    try:
        image = ImageOps.exif_transpose(image)
    except:
        pass

    if gray:
        image = image.convert("L")
        np_img = np.array(image)
    else:
        if image.mode == "RGBA":
            np_img = np.array(image)
            alpha_channel = np_img[:, :, -1]
            np_img = cv2.cvtColor(np_img, cv2.COLOR_RGBA2RGB)
        else:
            image = image.convert("RGB")
            np_img = np.array(image)

    if return_exif:
        return np_img, alpha_channel, exif_infos
    return np_img, alpha_channel


def norm_img(np_img):
    if len(np_img.shape) == 2:
        np_img = np_img[:, :, np.newaxis]
    np_img = np.transpose(np_img, (2, 0, 1))
    np_img = np_img.astype("float32") / 255
    return np_img


def resize_max_size(
    np_img, size_limit: int, interpolation=cv2.INTER_CUBIC
) -> np.ndarray:
    # Resize image's longer size to size_limit if longer size larger than size_limit
    h, w = np_img.shape[:2]
    if max(h, w) > size_limit:
        ratio = size_limit / max(h, w)
        new_w = int(w * ratio + 0.5)
        new_h = int(h * ratio + 0.5)
        return cv2.resize(np_img, dsize=(new_w, new_h), interpolation=interpolation)
    else:
        return np_img


def pad_img_to_modulo(
    img: np.ndarray, mod: int, square: bool = False, min_size: Optional[int] = None
):
    """

    Args:
        img: [H, W, C]
        mod:
        square: 是否为正方形
        min_size:

    Returns:

    """
    if len(img.shape) == 2:
        img = img[:, :, np.newaxis]
    height, width = img.shape[:2]
    out_height = ceil_modulo(height, mod)
    out_width = ceil_modulo(width, mod)

    if min_size is not None:
        assert min_size % mod == 0
        out_width = max(min_size, out_width)
        out_height = max(min_size, out_height)

    if square:
        max_size = max(out_height, out_width)
        out_height = max_size
        out_width = max_size

    return np.pad(
        img,
        ((0, out_height - height), (0, out_width - width), (0, 0)),
        mode="symmetric",
    )


def boxes_from_mask(mask: np.ndarray) -> List[np.ndarray]:
    """
    Args:
        mask: (h, w, 1)  0~255

    Returns:

    """
    height, width = mask.shape[:2]
    _, thresh = cv2.threshold(mask, 127, 255, 0)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        box = np.array([x, y, x + w, y + h]).astype(int)

        box[::2] = np.clip(box[::2], 0, width)
        box[1::2] = np.clip(box[1::2], 0, height)
        boxes.append(box)

    return boxes


def only_keep_largest_contour(mask: np.ndarray) -> List[np.ndarray]:
    """
    Args:
        mask: (h, w)  0~255

    Returns:

    """
    _, thresh = cv2.threshold(mask, 127, 255, 0)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    max_area = 0
    max_index = -1
    for i, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        if area > max_area:
            max_area = area
            max_index = i

    if max_index != -1:
        new_mask = np.zeros_like(mask)
        return cv2.drawContours(new_mask, contours, max_index, 255, -1)
    else:
        return mask
