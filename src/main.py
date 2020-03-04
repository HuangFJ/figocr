import logging
from model.model import OCRModel
from sketchs.ocr import OCR
import cv2
import argparse
from pathlib import Path

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, help='path to model')
    parser.add_argument('--image', required=True, help='path to image')

    opt = parser.parse_args()

    logging.root.setLevel(logging.INFO)
    model = OCRModel()

    if Path(opt.model).exists():
        model.load_checkpoint(opt.model)

    if Path(opt.image).exists():
        image = cv2.imread(opt.image)
        print(model.predict(image))
