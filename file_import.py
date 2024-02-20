import numpy as np
import matplotlib.pyplot as plt

import re

def read_xy_points(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()

    # Extract the XYPOINTS section using regular expression
    xy_points_match = re.search(r'##XYPOINTS=(.*?)\n(?:##|$)', content, re.DOTALL)
    if xy_points_match:
        xy_points_str = xy_points_match.group(1).strip()

        # Remove the '(XY..XY)' part
        xy_points_str = xy_points_str.replace('(XY..XY)', '').strip()
        # Check if XYPOINTS string is not empty
        if xy_points_str:
            # Extract X, Y values from each point and store in a 2D array
            xy_array = [list(map(float, re.split(r',', point.strip()))) for point in xy_points_str.split('\n') if point.strip()]
            return xy_array
        else:
            print("XYPOINTS section is empty.")
            return None
    else:
        print("XYPOINTS section not found in the file.")
        return None

def import_file(file_path):

    xy_array = read_xy_points(file_path)

    xy_array = np.array(xy_array, dtype=np.complex128)

    # convert phase (180 degree to negative sign)
    '''
    for n in range(len(xy_array)):
        if xy_array[n, 1] == 180:
            xy_array[n , 0] = -xy_array[n, 0]
    '''
    return xy_array

