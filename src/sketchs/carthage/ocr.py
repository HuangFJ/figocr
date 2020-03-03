# -*- coding: utf8 -*-
from sketchs.base import Base
from collections import OrderedDict
import logging
from pathlib import Path
import numpy as np
import subprocess
from xml.dom import minidom
import re
from procedure.image import Image, threshold, roi_detect, max_width_poly, cv2_imshow
import cv2


class OCR(Base):

    def __init__(self):
        self.data_dir = Path(__file__).parent.joinpath('data')
        self.model = None

    def load_model(self, model):
        self.model = model
    
    def parse_target_csv(self, csv_file):
        """
        {
            0: {'farm':'', 'week':'', 'date':''},
            1: { // row 1
                1:'', // column 1
                2:'', // column 2
                3:'', // column 3
                ...
            }, 
            2: { // row 2
                1:'', // column 1
                2:'', // column 2
                3:'', // column 3
                ...
            }, 
            ...
        }
        """
        result = {}
        with open(csv_file, 'r') as f:
            result[0] = dict(zip(['farm','week','date'], map(lambda x: x.strip(), f.readline().rstrip(' ,\n').split(','))))
            while True:
                line = f.readline()
                if not line:
                    break
                row_data = line.rstrip().split(',')
                row_num = int(row_data[0])
                other = row_data[1:]
                if other:
                    result[row_num] = {}
                    for i, cell in enumerate(other):
                        result[row_num][i+1] = cell.strip()
        return result

    def parse_frame_svg(self, svg_file):
        """
        {
            'farm': [x, y, width, height],
            'farm:expr': farm_expr,
            'week': [x, y, width, height],
            'week:expr': week_expr,
            'date': [x, y, width, height],
            'date:expr': date_expr,
            'y_scales': [y1, y2, ...],
            'x_scales': [x1, x2, ...],
            'x_scales:expr': [x1_expr, x2_expr, ...],
        }
        """
        result = {
            'farm': None,
            'farm:expr': None,
            'week': None,
            'week:expr': None,
            'date': None,
            'date:expr': None,
            'y_scales': None,
            'x_scales': None,
            'x_scales:expr': None,
        }
        root = minidom.parse(str(svg_file)).documentElement
        for node in root.childNodes:
            if node.nodeName == 'g':
                for node in node.childNodes:
                    if node.nodeName == 'rect':
                        rect = np.array([node.attributes['x'].value,
                                        node.attributes['y'].value,
                                        node.attributes['width'].value,
                                        node.attributes['height'].value]).astype(float).astype(int)
                        expr = node.attributes['id'].value
                        if 'farm' in expr:
                            key = 'farm'
                        elif 'week' in expr:
                            key = 'week'
                        elif 'date' in expr:
                            key = 'date'
                        else:
                            continue
                        result[key] = rect
                        result[f'{key}:expr'] = expr
                    elif node.nodeName == 'g':
                        identity = node.attributes['id'].value
                        scales = []
                        transform = np.array(node.attributes['transform'].value[10:-1].split(',')).astype(float).astype(int)
                        for node in node.getElementsByTagName('path'):
                            m=re.search(r'M(\d+),(\d+)', node.attributes['d'].value)
                            scales.append((
                                transform + np.array([m.group(1),m.group(2)]).astype(float).astype(int),
                                node.attributes['id'].value
                            ))
                        
                        if identity == 'horizon':
                            scales = [s[0][1] for s in scales]
                            scales.sort()
                            result['y_scales'] = scales
                        else:
                            scales.sort(key=lambda x: x[0][0])
                            result['x_scales'] = [s[0][0] for s in scales]
                            result['x_scales:expr'] = [s[1] for s in scales]

                break
        
        return result
    
    def guess_frame_name(self, image_file):
        out = b''
        with subprocess.Popen(["tesseract", f'{image_file}', "-", "-l", "eng"], stdout=subprocess.PIPE) as proc:
            proc.wait()
            out = proc.stdout.read()
            proc.stdout.close()

        if b'WEAN' in out:
            return 'CM-MWL-E-01'
        elif b'PIGLET' in out:
            return 'CM-PML-E-01'
        elif b'TREATMENT' in out:
            return 'CM-TEL-E-00'
        elif b'FARROW' in out:
            return 'CM-MFL-E-01'
        elif b'MASTER BREEDING' in out:
            return 'CM-MBL-E-01'
        elif b'STOCK' in out:
            return 'CM-BSR-E-00'
        elif b'BREEDING EVENTS' in out:
            return 'CM-BEL-E-00'
        else:
            return None

    def get_boxes(self, image, meta):
        """
        {
            0: {'farm':pt, 'week':pt, 'date':pt},
            1: { // row 1
                1:pt, // column 1
                2:pt, // column 2
                3:pt, // column 3
                ...
            }, 
            ...
        }
        pt is like :
        (x, y, width, height)
        """
        table = image[meta['y_scales'][0] : meta['y_scales'][-1], meta['x_scales'][0] : meta['x_scales'][-1]]
        thresh = threshold(table)

        farm = roi_detect(image, tuple(meta['farm']), thresh)
        week = roi_detect(image, tuple(meta['week']), thresh)
        date = roi_detect(image, tuple(meta['date']), thresh)

        table_boxes = OrderedDict({
            0: {'farm': farm, 'week': week, 'date': date}
        })

        for j in range(len(meta['y_scales'])-1):
            cell_y = meta['y_scales'][j]
            cell_height = meta['y_scales'][j + 1] - meta['y_scales'][j]
            
            row_rect = (int(meta['x_scales'][0]), int(cell_y), int(meta['x_scales'][-1] - meta['x_scales'][0]), int(cell_height))
            max_poly = max_width_poly(image, row_rect, thresh)
            if max_poly is None:
                continue

            # cv2.rectangle(image_copy, (max_poly[0], max_poly[1]), (max_poly[0]+max_poly[2], max_poly[1]+max_poly[3]), (0, 255, 0), 1)

            row = OrderedDict({})
            row_nul_cell = 0
            max_cell = 0
            for i in range(len(meta['x_scales'])-1):
                cell_x = meta['x_scales'][i]
                cell_width = meta['x_scales'][i + 1] - meta['x_scales'][i]
                rect = (int(cell_x), int(cell_y), int(cell_width), int(cell_height))
                row[i+1] = roi_detect(image, rect, thresh)
                
                if row[i+1] is None:
                    row_nul_cell += 1
                
                if cell_width > max_cell:
                    max_cell = cell_width
            
            if max_poly[2] > max_cell:
                continue
            if row_nul_cell == len(row):
                continue

            table_boxes[j+1] = row
        
        return table_boxes

    def deskew(self, image_file, frame_name=None):
        """
        afford raw image_file
        generate intermediate aligned image and return it
        """
        output_dir = Path(image_file).parent
        image_name = Path(image_file).name

        align_file = output_dir.joinpath(f'{image_name[:-4]}.align.png')
        if align_file.exists():
            return cv2.imread(str(align_file))

        if frame_name is None:
            frame_name = self.guess_frame_name(image_file)

        if not frame_name:
            return None

        frame_file = self.data_dir.joinpath(f'{frame_name}.png')

        image = Image.align_images(image_file, frame_file)
        if image is not None:
            cv2.imwrite(str(align_file), image)

        return image

    def training(self, dataset_dir, image_file, frame_name=None):
        """
        afford the raw image_file and corrected csv file with same name in the same folder
        generate dataset images and appended list in dataset_dir
        """
        output_dir = Path(image_file).parent
        image_name = Path(image_file).name

        svg_file = self.data_dir.joinpath(f'{frame_name}.svg')
        meta = self.parse_frame_svg(svg_file)

        dataset_images_dir = Path(dataset_dir).joinpath('images')
        dataset_list_file = Path(dataset_dir).joinpath('list.txt')
        image = self.deskew(image_file, frame_name)
        if image is None:
            logging.error(image_file)
            raise f'fail to process the input image {image_file}'

        dataset_list = []
        csv_file = output_dir.joinpath(f'{image_name[:-4]}.csv')
        target_data = self.parse_target_csv(csv_file)

        table_boxes = self.get_boxes(image, meta)

        csv_data = []
        for row_num, row in table_boxes.items():
            if row:
                row_data = []
                if row_num > 0:
                    row_data = [row_num]
                csv_data.append(row_data)

                for column, cell in row.items():
                    row_data.append('')
                    
                    if row_num > 0:
                        expr = meta['x_scales:expr'][column-1]
                        if expr.startswith('set') or expr.startswith('nil'):
                            continue
                    if cell:
                        roi_target = 255 - image[cell[1]: cell[1] + cell[3], cell[0]: cell[0] + cell[2]]

                        datum = target_data[row_num][column]
                        datum_file = f'{image_name}-{row_num}-{column}_{datum}.png'
                        cv2.imwrite(f'{dataset_images_dir}/{datum_file}', roi_target)
                        dataset_list.append(f'{datum_file}\t{datum}')
                        
        
        with open(dataset_list_file, 'a') as list_file:
            list_file.writelines(dataset_list)
        
    def recognize(self, image_file, frame_name=None):
        """
        afford the raw image_file 
        generate image with box and csv file, both with same name in the same folder
        """
        if self.model is None:
            raise f'you should load model first'

        output_dir = Path(image_file).parent
        image_name = Path(image_file).name

        svg_file = self.data_dir.joinpath(f'{frame_name}.svg')
        meta = self.parse_frame_svg(svg_file)

        box_file = output_dir.joinpath(f'{image_name[:-4]}.box.png')
        csv_file = output_dir.joinpath(f'{image_name[:-4]}.csv')

        image = self.deskew(image_file, frame_name)
        if image is None:
            logging.error(image_file)
            raise f'fail to process the input image {image_file}'
        image_copy = image.copy()
        
        table_boxes = self.get_boxes(image, meta)

        csv_data = []
        for row_num, row in table_boxes.items():
            if row:
                row_data = []
                if row_num > 0:
                    row_data = [row_num]
                csv_data.append(row_data)

                for column, cell in row.items():
                    row_data.append('')
                    
                    if row_num > 0:
                        expr = meta['x_scales:expr'][column-1]
                        if expr.startswith('set') or expr.startswith('nil'):
                            continue
                    if cell:
                        roi_target = 255 - image[cell[1]: cell[1] + cell[3], cell[0]: cell[0] + cell[2]]

                        cv2.rectangle(image_copy, (cell[0], cell[1]), (cell[0]+cell[2], cell[1]+cell[3]), (0, 0, 255), 1)
                        
                        pred = self.model.predict(roi_target)
                        row_data[-1]=pred
                        # cv2.putText(image_copy, pred, (cell[0], cell[1] - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
        
        # image_copy = cv2.resize(image_copy, None, fx=0.7, fy=0.7, interpolation = cv2.INTER_CUBIC)
        # cv2_imshow(image_copy)
        cv2.imwrite(str(box_file), image_copy)
        with open(csv_file, 'w') as csv:
            for row in csv_data:
                csv.write(','.join(map(lambda x:str(x), row)) + '\n')


if __name__ == '__main__':
    import os
    logging.root.setLevel(logging.INFO)

    dirs = [
        "/Users/jon/Documents/cv/data/CM-BEL-E-00",
        # "/Users/jon/Documents/cv/data/CM-BSR-E-00",
        # "/Users/jon/Documents/cv/data/CM-MBL-E-01",
        # "/Users/jon/Documents/cv/data/CM-MFL-E-01",
        # "/Users/jon/Documents/cv/data/CM-MWL-E-01",
        # "/Users/jon/Documents/cv/data/CM-PML-E-01",
        # "/Users/jon/Documents/cv/data/CM-TEL-E-00"
    ]
    image_num = 0
    for data_dir in dirs:
        template_name = data_dir[-11:]

        for _, _, filenames in os.walk(data_dir):

            for filename in filenames:
                image_num += 1
                if image_num > 2:
                    break

                image_file = f'{data_dir}/{filename}'
                ocr = OCR()
                ocr.load_model()
                ocr.recognize(image_file)
