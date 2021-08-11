import cv2
import numpy as np
import os
import csv
import math
from numpy import linalg as LA

def get_files(dirpath, ext):
    files = [s for s in os.listdir(dirpath)
         if os.path.isfile(os.path.join(dirpath, s)) and os.path.splitext(s)[1] == ext]
    files.sort()
    return files

def createDir(dirpath, ext):
    if os.path.exists(dirpath):
        for file in get_files(dirpath, ext):
            os.remove(os.path.join(dirpath, file))
    else:
        os.mkdir(dirpath)

def largestTrianglePointsIdxs(points):
    area = 0
    pointsIdxs = [0,0,0]
    if len(points) < 3:
        return None
    
    for i in range(len(points)-2):
        x1,y1 = points[i]
        for j in range(i+1,len(points)-1):
            x2,y2 = points[j]
            for k in range(j+1,len(points)):
                x3,y3 = points[k]
                if abs(0.5*(x1*(y2-y3)+x2*(y3-y1)+x3*(y1-y2))) > area :
                    area = abs(0.5*(x1*(y2-y3)+x2*(y3-y1)+x3*(y1-y2)))
                    pointsIdxs = [i,j,k]

    return pointsIdxs

def findNearesPoints(points, keypoints):
    nearest_points = []
    for point in points:
        minDist = MIN_DIST
        minKeypoint = None
        for keypoint in keypoints:
            if math.sqrt((point[0] - keypoint[0])**2 + (point[1] - keypoint[1])**2) < minDist:
                minDist = math.sqrt((point[0] - keypoint[0])**2 + (point[1] - keypoint[1])**2)
                minKeypoint = keypoint
                
        if minKeypoint is None:
            return None
        
        nearest_points.append(minKeypoint)
        
    return nearest_points

def alignImages(table_file, regions_pattern, regions_table, 
                original_image, rotate_matrix, border_points):
    keypoints_pattern = regions_pattern
    keypoints_table = regions_table
    
    print(len(keypoints_pattern), len(keypoints_table))
    if len(keypoints_pattern) < 3 or len(keypoints_table) < 3:
        print("Not enough elements")
        cv2.imwrite(RESULTS_DIR + table_file + "_borders.jpg", original_image)
        return

    pts1 = []
    pts2 = []
    new_matches = []
    for pIndex, keypoint_pattern in enumerate(keypoints_pattern):
        minDist = MIN_DIST
        pairIndex = 0
        for tIndex, keypoint_table in enumerate(keypoints_table):
            if math.sqrt((keypoint_pattern[0] - keypoint_table[0])**2 + (keypoint_pattern[1] - keypoint_table[1])**2) < minDist:
                if keypoint_table not in pts2:
                    pairIndex = tIndex
                    minDist = math.sqrt((keypoint_pattern[0] - keypoint_table[0])**2 
                                      + (keypoint_pattern[1] - keypoint_table[1])**2)
        if minDist < MIN_DIST:
            keypoint_table = keypoints_table[pairIndex]
            dmatch = cv2.DMatch(pIndex, pairIndex, 100)
            cv2.circle(original_image, (int(keypoint_pattern[0]), int(keypoint_pattern[1])), 
                       radius=MIN_DIST, color=(255, 0, 0), thickness=2)
            new_matches.append(dmatch)
            pts1.append(keypoint_pattern)
            pts2.append(keypoint_table)
        
    pts1 = np.float32(pts1)
    pts2 = np.float32(pts2)
            
    for point in keypoints_pattern:
        cv2.circle(original_image, (int(point[0]), int(point[1])), radius=10, color=(255, 0, 0), thickness=-1)
        
    for point in keypoints_table:
        cv2.circle(original_image, (int(point[0]), int(point[1])), radius=10, color=(0, 0, 255), thickness=-1)
        
    for point in pts2.astype(int):
        cv2.circle(original_image, (int(point[0]), int(point[1])), radius=10, color=(0, 255, 0), thickness=-1)
    
    print("Matches:", len(new_matches))
    if len(new_matches) < 3:
        print("Lack of matches")
        cv2.imwrite(RESULTS_DIR + table_file + "_borders.jpg", original_image)
        return 

    pointsIndexes = largestTrianglePointsIdxs(pts1)
    pts1_3 = np.float32([pts1[pointsIndexes[0]], pts1[pointsIndexes[1]], pts1[pointsIndexes[2]]])
    pts2_3 = np.float32([pts2[pointsIndexes[0]], pts2[pointsIndexes[1]], pts2[pointsIndexes[2]]])
    matrixAff = cv2.getAffineTransform(pts1_3, pts2_3) 
        
    pts_for_affin =  np.array([pts1_3], np.float32)
    dst_affin = cv2.transform(pts_for_affin, matrixAff).astype(int)
            
    fields_borders = []
    
    if len(new_matches) > len(keypoints_table) * 0.8:
        
        for point in dst_affin[0]:
            cv2.circle(original_image, (int(point[0]), int(point[1])), radius=10, color=(255, 0, 255), thickness=-1)
        
        cv2.line(original_image, (dst_affin[0][0][0], dst_affin[0][0][1]), 
                 (dst_affin[0][1][0], dst_affin[0][1][1]), 
                 (255, 0, 255), 3, cv2.LINE_AA)
        cv2.line(original_image, (dst_affin[0][1][0], dst_affin[0][1][1]),
                 (dst_affin[0][2][0], dst_affin[0][2][1]),
                 (255, 0, 255), 3, cv2.LINE_AA)
        cv2.line(original_image, (dst_affin[0][2][0], dst_affin[0][2][1]),
                 (dst_affin[0][0][0], dst_affin[0][0][1]),
                 (255, 0, 255), 3, cv2.LINE_AA)        
        
        for b_dict in border_points:
            field_type = b_dict['field_type']
            pts_border = b_dict['coords'].copy()
            pts_border = np.array(pts_border, np.int)
            
            dst_keypoints = findNearesPoints(pts_border, pts2)
            if dst_keypoints is None:
                pts_border_inv = np.array([[pts_border[0][0], pts_border[1][1]],
                                           [pts_border[1][0], pts_border[0][1]]], np.float32)

                pts_border_inv = np.array(pts_border_inv, np.float32)

                dst_keypoints_inv = findNearesPoints(pts_border_inv, pts2)
                print(dst_keypoints)
                if dst_keypoints is None:
                    pts_border_np =  np.array([pts_border], np.float32)
                    dst_border = cv2.transform(pts_border_np, matrixAff).astype(int)[0]
                else:
                    dst_border = np.array(dst_keypoints_inv.copy(), np.int) 
            else:
                dst_border = np.array(dst_keypoints.copy(), np.int)    

            original_image = cv2.rectangle(original_image, (dst_border[0][0], dst_border[0][1]),
                                (dst_border[1][0], dst_border[1][1]), (0, 255, 0), 5)
            
            fields_borders.append({'field_type': field_type, 'coords' : dst_border})

    #     inv_rotate_matrix = cv2.invertAffineTransform(rotate_matrix)
    #     dst_border = cv2.transform(dst_border, inv_rotate_matrix).astype(int)
    
    cv2.imwrite(RESULTS_DIR + table_file + "_borders.jpg", original_image)
    return fields_borders

def rotateIMG(image):
    midImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
    BLACK_LIMIT = 200
    
    midImage[midImage < BLACK_LIMIT] = 0
    midImage[midImage > BLACK_LIMIT] = 255

    thresh = cv2.threshold(midImage, 190, 255, cv2.THRESH_BINARY_INV)[1]

    midImage = np.copy(thresh)
    
    dstImage = cv2.Canny(midImage, 80, 200, 3)
    
    lineimage = image.copy()
 
    lines = cv2.HoughLines(dstImage, 1, np.pi/180, 375)
    
    approvedCount = 0
    
    verticalCount = 0
    verticalSum = 0
    
    horizontalCount = 0
    horizontalSum = 0
    
    
    if lines is None:
        print("No lines")
        return
    
    for i in range(len(lines)):
        for rho, theta in lines[i]:
            lineAngle = theta / np.pi * 180
            
            if (lineAngle > 180):
                print(lineAngle)
                
            if (lineAngle < 80 or (lineAngle > 100 and lineAngle < 170)                 or (lineAngle > 190 and lineAngle < 260) or lineAngle > 280):
                continue
            else:
                horizontalCount += 1
                horizontalSum += lineAngle   
                
                            
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a * rho
                y0 = b * rho
                x1 = int(round(x0 + 4000 * (-b)))
                y1 = int(round(y0 + 4000 * a))
                x2 = int(round(x0 - 4000 * (-b)))
                y2 = int(round(y0 - 4000 * a))

    average = horizontalSum / horizontalCount - 90 if horizontalCount > 0 else 0    
    rotate = lineimage
        
    h, w = rotate.shape[:2]
    RotateMatrix = cv2.getRotationMatrix2D((w/2.0, h/2.0), average, 1)
    rotateImg = cv2.warpAffine(rotate, RotateMatrix, (w, h), borderValue=(255, 255, 255))

    return rotateImg, RotateMatrix


def line_intersection(l1, l2):
    line1 = ([l1[0],l1[1]],[l1[2],l1[3]])
    line2 = ([l2[0],l2[1]],[l2[2],l2[3]])
    
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
        return None

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return (int(x), int(y))


def distance(point1, point2):
    return math.sqrt((point1[0]-point2[0])**2 + (point1[1]-point2[1])**2)


def isPontOnLine(point, line):
    return abs(distance([line[0],line[1]], point) + distance([line[2],line[3]], point)         - distance([line[0],line[1]], [line[2],line[3]])) < 3


def getKeyPoints(rotate_img, saveJpeg = False):
    global pcount 
    midImage = cv2.cvtColor(rotate_img, cv2.COLOR_BGR2GRAY)
            
    BLACK_LIMIT = 200
    
    midImage[midImage < BLACK_LIMIT] = 0
    midImage[midImage > BLACK_LIMIT] = 255

    thresh = cv2.threshold(midImage, 170, 200, cv2.THRESH_BINARY_INV)[1]

    horizontal = np.copy(thresh)
    vertical = np.copy(thresh)
    
    scale = 10
        
    horizontalCols = horizontal.shape[1]
    horizontalSize = horizontalCols / scale
    horizontalSize = int(horizontalSize)
    horizontalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontalSize, 1))
    horizontal = cv2.morphologyEx(horizontal, cv2.MORPH_OPEN, horizontalStructure)

    
    verticalRows = vertical.shape[0]
    verticalSize = verticalRows / scale
    verticalSize = int(verticalSize)
    verticalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (1, verticalSize))
    vertical = cv2.morphologyEx(vertical, cv2.MORPH_OPEN, verticalStructure)
    
    
    SE = cv2.getStructuringElement(cv2.MORPH_RECT, (1,5))
    horizontal = cv2.dilate(horizontal, SE, iterations=1)

    SE = cv2.getStructuringElement(cv2.MORPH_RECT, (5,1))
    vertical = cv2.dilate(vertical, SE, iterations=1)
    
    lines_mask = horizontal + vertical

    # cv2.imwrite(RESULTS_DIR + "im" + str(pcount) + "_linesmask.jpg", lines_mask)
    
    
    points_image = rotate_img.copy()
    
    linesP = cv2.HoughLinesP(lines_mask, 1, np.pi/180, 50, None, 20, 20)
    # linesP = cv2.HoughLines(lines_mask, 1, np.pi / 180, 150)
    

    if linesP is not None:    
        horizontalLines = [line[0] for line in linesP if abs(line[0][2] - line[0][0]) > abs(line[0][3] - line[0][1])]
        verticalLines = [line[0] for line in linesP if abs(line[0][2] - line[0][0]) < abs(line[0][3] - line[0][1])]
        
        LINE_OFFSET = 20
        count = 0
        for i in range(0, len(horizontalLines)):
            l = horizontalLines[i]
            l[0] -= LINE_OFFSET
            l[2] += LINE_OFFSET
            horizontalLines[i] = l
            # points_image = cv2.line(points_image, (l[0], l[1]), (l[2], l[3]), (0, 0, 255), 2, cv2.LINE_AA)
            # cv2.imwrite(RESULTS_DIR + table_file + str(count) + "_points.jpg", lineimg)
            count += 1
            
        for i in range(0, len(verticalLines)):
            l = verticalLines[i]
            l[1] += LINE_OFFSET
            l[3] -= LINE_OFFSET
            verticalLines[i] = l
            # points_image = cv2.line(points_image, (l[0], l[1]), (l[2], l[3]), (0, 0, 255), 2, cv2.LINE_AA)
            # cv2.imwrite(RESULTS_DIR + table_file + str(count) + "_points.jpg", lineimg)
            count += 1
            
        keypoints = []
        count = 0
        for horizontal_line in horizontalLines:
            for vertical_line in verticalLines:
                point = line_intersection(horizontal_line, vertical_line)
                if point is not None:
                    if isPontOnLine(point, vertical_line) and isPontOnLine(point, horizontal_line):
                        keypoints.append(point)
                    
        hasChanges = True
        while hasChanges:
            hasChanges = False
            
            for keypoint in keypoints:
                if hasChanges:
                    break
                neighbours = []
                for neighbour in keypoints:
                    if keypoint == neighbour:
                        continue
                        
                    if math.sqrt((keypoint[0]-neighbour[0])**2 + (keypoint[1]-neighbour[1])**2) < 30:
                        neighbours.append(neighbour)

                if len(neighbours) > 0:
                    new_keypoint = keypoint
                    for neighbour in neighbours:
                        new_keypoint = (new_keypoint[0] + neighbour[0], 
                                        new_keypoint[1] + neighbour[1])
                    
                    new_keypoint = (int(new_keypoint[0] / (len(neighbours) + 1)), 
                                    int(new_keypoint[1] / (len(neighbours) + 1)))

                    keypoints = [point for point in keypoints if point not in neighbours and point != keypoint]
                    keypoints.append(new_keypoint)
                    
                    hasChanges = True
                    break
        
        im_heigth, im_width = points_image.shape[:2]
        PIXEL_OFFSET = 20
        x_left = max(0, min([point[0] for point in keypoints]) - PIXEL_OFFSET)
        x_right = min(im_width - 1, max([point[0] for point in keypoints]) + PIXEL_OFFSET)
        y_top = max(0, min([point[1] for point in keypoints]) - PIXEL_OFFSET)
        y_bottom = min(im_heigth - 1, max([point[1] for point in keypoints]) + PIXEL_OFFSET)

        # cv2.rectangle(points_image, (x_left, y_top),
        #                         (x_right, y_bottom), (0, 255, 0), 5)

        cropped_im = points_image[y_top:y_bottom, x_left:x_right]

        for index in range(len(keypoints)):
            point = keypoints[index]
            point = (point[0] - x_left, point[1] - y_top)
            keypoints[index] = point
            cv2.circle(cropped_im, point, radius=10, color=(0, 255, 0), thickness=-1)

                    
        if saveJpeg:
            cv2.imwrite(RESULTS_DIR + "im" + str(pcount) + "_points.jpg", cropped_im)
        
        return cropped_im, keypoints, x_left, y_top


MIN_DIST = 35
RESULTS_DIR = "results/"
PATTERN_PATH = "patterns/"

pcount = 0
createDir(RESULTS_DIR, '.jpg')

def match_pattern(table_im):
    global pcount
    pcount += 1

    pattern_names = get_files(PATTERN_PATH, '.jpg')

    # cv2.imwrite(RESULTS_DIR + "im" + str(pcount) + "_original.jpg", table_im)
        
    rotated_table_im, rotate_matrix = rotateIMG(table_im.copy())
    cropped_rotated_table_im, keypoints_table, offset_x, offset_y = getKeyPoints(rotated_table_im)
    
    for pattern_name in pattern_names:
        pattern_im = cv2.imread(PATTERN_PATH + pattern_name, cv2.IMREAD_COLOR)
        rotate_pattern, _ = rotateIMG(pattern_im)
        rotate_pattern, keypoints_pattern, _, _ = getKeyPoints(rotate_pattern)

        border_points = []
        with open(PATTERN_PATH + pattern_name[:-4] + '.csv') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                border_points.append({'field_type': row['field_type'],
                    'coords': [[row['x0'],row['y0']],[row['x1'],row['y1']]]})

        if keypoints_table is not None \
            and abs(len(keypoints_pattern) - len(keypoints_table)) < len(keypoints_pattern) * 0.15:

            borders = alignImages("img", keypoints_pattern.copy(), 
                        keypoints_table.copy(), cropped_rotated_table_im.copy(), 
                        rotate_matrix.copy(), border_points)

            if borders is not None:
                for index in range(len(borders)):
                    border = borders[index]
                    coords = border['coords']
                    coords[0][0] += offset_x
                    coords[0][1] += offset_y
                    coords[1][0] += offset_x
                    coords[1][1] += offset_y
                    border['coords'] = coords
                    borders[index] = border

                return borders
            else:
                print("Incorrect pattern, next pattern")
        else:
            print("None regions, next pattern")


# In[ ]:





# In[ ]:





# In[ ]:




