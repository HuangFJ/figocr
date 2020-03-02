# -*- coding: utf8 -*-
import cv2
import numpy as np
from skimage.morphology import opening, closing, square
from skimage.filters import threshold_otsu
import logging


def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image

    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)

    else:
        r = width / float(w)
        dim = (width, int(h * r))

    resized = cv2.resize(image, dim, interpolation = inter)

    return resized


def cv2_imshow(image):
    import matplotlib.pyplot as plt
    plt.imshow(image, interpolation = 'bicubic')
    plt.show()
    # cv2.imshow('debug', image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # cv2.waitKey(1)


def roi_detect(image, region, thresh_mean=None, trace=False):
    # (x, y, w, h)
    x, y, width, height = region
    
    up_y_offset = int(height / 2)
    up_y = y - up_y_offset
    down_y_offset = height + int(height / 2)
    down_y = y + down_y_offset

    roi = image[up_y : down_y, x : x + width]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    
    try:
        thresh_value = int(1*threshold_otsu(gray))
    except:
        thresh_value = 255

    if thresh_mean is not None:
        thresh_value = min(thresh_value, thresh_mean)
    thresh = cv2.threshold(gray, thresh_value, 255, cv2.THRESH_BINARY_INV)[1]
    thresh = closing(thresh)
    #垂直计算留白
    yitensity = np.sum(thresh, axis=1)

    middle = height
    step = 1
    while yitensity[middle] == 0:
        middle = height + step
        if step < 0:
            step = abs(step)+1
        else:
            step = - step

        if abs(step) > 20:
            # 行中间水平线上下20个单位内没有内容
            return None

    # 上边距4个空白单位
    up_blank_line = 0
    for i in reversed(range(middle)):
        if yitensity[i] == 0:
            up_y_offset = i
            up_blank_line += 1
        if up_blank_line > 3:
            break
    # 下边距4个空白单位
    down_blank_line = 0
    for i in range(middle, (down_y - up_y)):
        if yitensity[i] == 0:
            down_y_offset = i
            down_blank_line += 1
        if down_blank_line > 3:
            break
    
    y = up_y + up_y_offset
    height = down_y_offset - up_y_offset

    # 垂直裁剪
    thresh = thresh[up_y_offset : down_y_offset, 0 : width]
    thresh = cv2.Canny(thresh, 100, 200, 3, L2gradient=True)
    thresh = cv2.dilate(thresh, None)
    # 水平计算留白
    xitensity = np.sum(thresh, axis=0)

    x_offset = 0
    x_suffix = len(xitensity) - 1
    while True:
        if (x_offset >= x_suffix) or (xitensity[x_offset] and xitensity[x_suffix]):
            break

        if xitensity[x_offset] == 0:
            x_offset += 1
        if xitensity[x_suffix] == 0:
            x_suffix -= 1

    x_offset = x_offset - 5 if x_offset - 5 > 0 else 0
    x_suffix = x_suffix + 5 if x_suffix + 5 < len(xitensity) else (len(xitensity) - 1)

    x = x + x_offset
    width = x_suffix - x_offset

    if height < 16 or width <= 10:
        # 行内容高度只有8个单位（小数点的大小）
        return None

    # # 水平裁剪
    # thresh = thresh[0 : height, x_offset : x_suffix]

    # cv2.rectangle(image, (x+cnt_x, y+cnt_y), (x+cnt_x + cnt_w-2, y+cnt_y + cnt_h-2), (0, 0, 0), 1)
    # cv2.rectangle(image, (x, y), (x + width-2, y + height-2), (0, 255, 0), 1)
    return x, y, width, height
    

def max_width_poly(image, region, thresh_mean=None):
    x, y, width, height = region
    
    roi = image[y : y + height, x : x + width]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    
    try:
        thresh_value = int(1*threshold_otsu(gray))
    except:
        thresh_value = np.percentile(gray, 50)
    if thresh_mean is not None:
        thresh_value = min(thresh_value, thresh_mean)
    thresh = cv2.threshold(gray, thresh_value, 255, cv2.THRESH_BINARY_INV)[1]

    thresh = cv2.Canny(thresh, 100, 200, 3, L2gradient=True)
    thresh = cv2.dilate(thresh, None)

    cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    boxes = map(lambda cnt: cv2.boundingRect(cnt), cnts)

    boxes = sorted(boxes, key=lambda x: x[2])
    if boxes:
        box = boxes.pop()
        return x+box[0], y+box[1], box[2], box[3]
    else:
        return None


def threshold(image):
    return int(1*threshold_otsu(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)))


def consine(pt1, pt2, pt0):
    v1 = np.array(pt1) - np.array(pt0)
    v2 = np.array(pt2) - np.array(pt0)
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))


def angle_degree(pt1, pt2, pt0):
    radian = np.arccos(consine(pt1, pt2, pt0))
    return np.degrees(radian)


def square_contours_kps(image, min_area=1800, min_density=None):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    edges = cv2.Canny(gray, 0, 0)

    # linesP = cv2.HoughLinesP(edges, 1, (np.pi / 180)*10, 50, None, 100, 10)
    # if linesP is not None:
    #     for i in range(0, len(linesP)):
    #         l = linesP[i][0]
    #         cv2.line(gray, (l[0], l[1]), (l[2], l[3]), (0,0,255), 1, cv2.LINE_AA)
    # cv2_imshow(gray)
    
    edges = closing(edges, square(3))
    # cv2_imshow(edges)
    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    squares = []
    for cnt in cnts:
        approx = cv2.approxPolyDP(cnt, cv2.arcLength(cnt, True)*0.02, True)
        area = cv2.contourArea(approx)

        # cv2.drawContours(image, [approx], -1, (0,0,255), 1) 

        if area < min_area:
            continue

        # cv2.drawContours(image, [approx], -1, (0,0,255), 1) 
        # print(f'area: {area}, approx: {approx}')

        # for v in approx:
        #     cv2.circle(image, tuple(v[0]), 3, (0, 255, 255), -1)

        vertex_num = len(approx)
        if vertex_num >= 4: #4个或多于4个顶点 and cv2.isContourConvex(approx):
            for offset in range(0, vertex_num): #取连续的4个顶点
                square_approx = []
                for idx in range(0, 4):
                    sq_index = (offset+idx) % vertex_num
                    square_approx.append(approx[sq_index])

                square_approx = np.array(square_approx)
                maxCosine = 0
                for j in range(2, 5):
                    square_pts = np.squeeze(square_approx)
                    cosine = abs(consine(square_pts[j%4], square_pts[j-2], square_pts[j-1]))
                    # print(cosine)
                    maxCosine = max(maxCosine, cosine)

                if maxCosine < 0.20: # up and down 12 degree
                    if vertex_num > 4:
                        area = cv2.contourArea(square_approx)
                        if area < min_area:
                            continue

                    mask = np.zeros(edges.shape, dtype="uint8")
                    cv2.drawContours(mask, [square_approx], -1, 255, -1)
                    mask = cv2.bitwise_and(edges, edges, mask=mask)
                    # cv2_imshow(mask)
                    mass = cv2.countNonZero(mask)
                    density = mass / area
                    logging.info(f'area:{area}, mass:{mass}')
                    if min_density is not None:
                        if density < min_density:
                            continue

                    squares.append((area, density, square_approx))
                    break

    # cv2.drawContours(image, np.array([sq[1] for sq in squares]), -1, (0,0,255), 1) 
    # cv2_imshow(image)
    if len(squares) < 4:
        return None
    
    # sort by area
    squares.sort(key=lambda sq: -sq[0])
    squares = squares[:4]

    result = []
    # (area, density, contour)
    for _, _, sq in squares:
        M = cv2.moments(sq)

        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])

        result.append([cx, cy])

    return result


def block_contours_kps(image, min_area=1800):
    # convert it to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # cv2_imshow(gray)

    result = None
    delta = 2
    while result is None and delta < 5: #O(3)
        # blur it slightly
        ksize = 2 * delta + 1
        blur = cv2.GaussianBlur(gray, (ksize, ksize), delta)
        # cv2_imshow(blur)

        # threshold the image 
        thresh_value = threshold_otsu(blur)
        thresh_factor = 3

        while result is None and thresh_factor > 0: #O(3*3)
            thresh = cv2.threshold(blur, thresh_value/thresh_factor, 255, cv2.THRESH_BINARY_INV)[1]
            # perform a series of erosions + dilations to remove any small regions of noise
            thresh = cv2.erode(thresh, None, iterations=2)
            thresh = cv2.dilate(thresh, None, iterations=2)
            # cv2_imshow(thresh)

            # find contours in thresholded image
            cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # image_copy = image.copy()
            # cv2.drawContours(image_copy, cnts, -1, (0,0,255), 2) 
            # cv2_imshow(image_copy)
            
            # 没有连通域形状
            if len(cnts) < 4:
                thresh_factor -= 1
                continue
            
            cnts = [(cnt, cv2.moments(cnt)) for cnt in cnts]
            cnts = [cnt for cnt in cnts if cnt[1]['m00'] > min_area/delta]
            # image_copy = image.copy()
            # cv2.drawContours(image_copy, [cnt[0] for cnt in cnts], -1, (0,255,0), 1) 
            # cv2_imshow(image_copy)

            # 找到形状面积超过设定值的形状
            if len(cnts) < 4:
                thresh_factor -= 1
                continue

            cnts = sorted(cnts, key=lambda cnt: -cnt[1]['m00'])
            # 黑色面积最大的四个形状
            cnts_target = cnts[:4]

            result = []
            for cnt, M in cnts_target:
                cx = int(M['m10']/M['m00'])
                cy = int(M['m01']/M['m00'])

                # M['m00'] is the mass, [cx, cy] is the centroid
                result.append([cx, cy])

            # image_copy = image.copy()
            # for index, (cnt, M) in enumerate(cnts_target):
            #     print(M['m00'])
            #     cv2.drawContours(image_copy, [cnt], -1, (0,255,0), 1) 
            #     cv2.circle(image_copy, tuple(result[index]), 5, (255, 255, 255), -1)
            # cv2_imshow(image_copy)

        delta += 1
    
    logging.info(f'delta: {delta}, thresh factor: {thresh_factor}')
    return result


def get_square_vertex(kps):
    """ 
    pt: [x, y]
    kps: [ pt1, pt2, pt3, pt4 ]
    return: [ top, right, bot, left ]
    """
    if not kps or len(kps) != 4:
        return None

    kps.sort(key=lambda p: p[0])
    left_p = kps[:2]
    right_p = kps[2:]

    left_p.sort(key=lambda p: p[1])
    extTop, extLeft = left_p
    right_p.sort(key=lambda p: p[1])
    extRight, extBot = right_p

    # rows,cols,_ = image.shape

    # angle_horizon = np.arctan2(extRight[1] - extTop[1], extRight[0] - extTop[0])
    # deg = np.rad2deg(angle_horizon)

    # M = cv2.getRotationMatrix2D((cols/2,rows/2),deg,1)
    # dst = cv2.warpAffine(image,M,(cols,rows))
    # cv2_imshow(dst)
    degrees = [
        angle_degree(extRight, extLeft, extTop), 
        angle_degree(extTop, extBot, extRight), 
        angle_degree(extRight, extLeft, extBot), 
        angle_degree(extBot, extTop, extLeft)
    ]
    degree_max = max(degrees)
    degree_min = min(degrees)
    # print(degree)
    if 89 <= degree_min and degree_max <= 91:
        return np.array([extTop, extRight, extBot, extLeft])
    else:
        return None


def draw_match_2_side(img1, kp1, img2, kp2, N):
    """Draw matches on 2 sides
    Args:
        img1 (HxW(xC) array): image 1
        kp1 (Nx2 array): keypoint for image 1
        img2 (HxW(xC) array): image 2
        kp2 (Nx2 array): keypoint for image 2
        N (int): number of matches to draw
    Returns:
        out_img (Hx2W(xC) array): output image with drawn matches
    """
    kp_list = np.linspace(0, min(kp1.shape[0], kp2.shape[0])-1, N, dtype=np.int)

    # Convert keypoints to cv2.Keypoint object
    cv_kp1 = [cv2.KeyPoint(x=pt[0], y=pt[1], _size=1) for pt in kp1[kp_list]]
    cv_kp2 = [cv2.KeyPoint(x=pt[0], y=pt[1], _size=1) for pt in kp2[kp_list]]

    out_img = np.array([])
    good_matches = [cv2.DMatch(_imgIdx=0, _queryIdx=idx, _trainIdx=idx,_distance=0) for idx in range(N)]
    out_img = cv2.drawMatches(img1, cv_kp1, img2, cv_kp2, matches1to2=good_matches, outImg=out_img)

    return out_img 


class Image(object):

    frames = {}

    @classmethod
    def get_frame(cls, frame_file):
        frame_file = str(frame_file)
        frame = cls.frames.get(frame_file)
        if frame is None:
            img = cv2.imread(frame_file)
            vertexs = get_square_vertex(block_contours_kps(img))
            if vertexs is None:
                return None
            frame = (img, vertexs)
            cls.frames[frame_file] = frame

        return frame
    
    @classmethod
    def get_image(cls, image_file):
        image_file = str(image_file)
        img = cv2.imread(image_file)
        vertexs = get_square_vertex(block_contours_kps(img))
        if vertexs is None:
            vertexs = get_square_vertex(square_contours_kps(img))

        if vertexs is not None:
            return (img, vertexs)
        else:
            return None

    @classmethod
    def align_images(cls, image_file, frame_file):
        # scanned image
        image_info = cls.get_image(image_file)
        if not image_info:
            return None
        img, kps = image_info

        # template image
        mask_info = cls.get_frame(frame_file)
        if not mask_info:
            return None
        mask_img, mask_kps = mask_info
        
        # Draw top matches
        image = draw_match_2_side(img, kps, mask_img, mask_kps, 4)

        ####### DEBUG #########
        # # resize image
        # image = cv2.resize(image, None, fx=0.5, fy=0.5, interpolation = cv2.INTER_CUBIC)
        # cv2_imshow(image)
        ####### DEBUG #########

        # Find homography
        m, mask = cv2.findHomography(kps, mask_kps, cv2.RANSAC, 5.0)

        # Use homography to warp image
        h, w, _ = mask_img.shape
        result = cv2.warpPerspective(img, m, (w, h))

        return result


if __name__ == '__main__':
    logging.root.setLevel(logging.INFO)
    
    bads = [
        '/Users/jon/Documents/cv/data/CM-MBL-E-01/OCR20170828_0020.tif',
        '/Users/jon/Documents/cv/data/CM-MBL-E-01/OCR2017091_0010.tif'
    ]
    goods = [
        '/Users/jon/Documents/cv/data/CM-MBL-E-01/CCE2017068_0008.tif',
        '/Users/jon/Documents/cv/data/CM-MBL-E-01/CCE2017068_0009.tif'
    ]
    for i in bads:
        image, vertexs = Image.get_image(i)
        extTop, extRight, extBot, extLeft = vertexs
        ####### DEBUG #########
        cv2.circle(image, tuple(extLeft), 2, (0, 0, 255), -1)
        cv2.circle(image, tuple(extRight), 2, (0, 255, 0), -1)
        cv2.circle(image, tuple(extTop), 2, (255, 0, 0), -1)
        cv2.circle(image, tuple(extBot), 2, (255, 255, 0), -1)
        cv2_imshow(image)
        ####### DEBUG #########