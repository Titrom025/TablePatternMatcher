import cv2
import os
import pattern_matcher as pm


RESULTS_DIR = "results/"
TABLES_DIR = "tables/"


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

createDir(RESULTS_DIR, '.jpg')

if not os.path.exists(TABLES_DIR):
    os.mkdir(TABLES_DIR)
  
tables_list = get_files("tables/", ".jpg")

if len(tables_list) == 0:
    raise Exception("Empty input folder! Place the table files in the input folder")

for table_file in tables_list:

    print(table_file)
    
    table_path = "tables/" + table_file
    table_im = cv2.imread(table_path, cv2.IMREAD_COLOR)

    try:
        pattern_boxes = pm.match_pattern(table_im.copy(), table_file[:-4], False, True)
        box_im = table_im.copy()
        if pattern_boxes is not None:
            print("Pattern found!")
            for box in pattern_boxes:
                field_type = box['field_type']
                coords = box['coords']
                print("Field: " + field_type + ", x0: " + str(coords[0][0]) + ", y0: " \
                    + str(coords[0][1]) + ", x1: " + str(coords[1][0]) + ", y1: " + str(coords[1][1]))
                cv2.rectangle(box_im, (coords[0][0], coords[0][1]), (coords[1][0], coords[1][1]), (0, 255, 0), 5)
        cv2.imwrite(RESULTS_DIR + "/" + table_file[:-4] + "_pattern.jpg", box_im)
    except Exception as e:
        print("Patern matching error:" + str(e))


    
    


