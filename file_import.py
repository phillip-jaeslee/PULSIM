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

    xy_array = np.array(xy_array, dtype=float)

    # convert phase (180 degree to negative sign)
    '''
    for n in range(len(xy_array)):
        if xy_array[n, 1] == 180:
            xy_array[n , 0] = -xy_array[n, 0]
    '''
    return xy_array

def equ2shape(equation_str, x_range=(0, 1), num_points=1000):
    
    ## string equation to shape
    """
    equ2shape(equation_str, x_range=(0, 1), num_points=1000)
    input:
    equation_str        - equation string
    optional:
    x_range             - range of x (default = (0, 1))
    num_points          - the number of points
    output:

    """ 

    # Extract the right-hand side of the equation
    match = re.match(r"y\s*=\s*(.+)", equation_str.replace(" ", ""))
    if not match:
        raise ValueError("Equation must be in the form 'y=...'")

    expr = match.group(1)

    # --- Preprocess expression ---
    # Replace '^' with '**'
    expr = re.sub(r'\^', '**', expr)

    # Insert * between number and variable: 2x → 2*x
    expr = re.sub(r'(\d)([a-zA-Z])', r'\1*\2', expr)

    # Insert * between variable and '(', but NOT when it's a known function like sin, cos
    functions = ['sin', 'cos', 'tan', 'exp', 'log', 'sqrt', 'abs']
    pattern = rf'\b(?!{"|".join(functions)})([a-zA-Z])\('
    expr = re.sub(pattern, r'\1*(', expr)

    print(f"Processed expression: {expr}")

    # Allowed functions
    allowed_names = {
        'sin': np.sin,
        'cos': np.cos,
        'tan': np.tan,
        'exp': np.exp,
        'log': np.log,
        'sqrt': np.sqrt,
        'abs': np.abs,
        'pi': np.pi,
        'e': np.e,
        'x': None
    }

    # Generate x values
    x = np.linspace(x_range[0], x_range[1], num_points)
    allowed_names['x'] = x

    # Evaluate safely
    try:
        y = eval(expr, {"__builtins__": {}}, allowed_names)
    except Exception as e:
        raise ValueError(f"Error evaluating expression: {e}")

    # Plot
    plt.plot(x, y)
    plt.title(f"Plot of {equation_str}")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid(True)
    plt.show()


equation="y=sin(x/2pi)^2 + cos(x)"

# Example usage:
#equ2shape(equation, x_range=(0, np.pi))

