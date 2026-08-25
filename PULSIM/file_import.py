import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
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

def equ2shape(equation_str, x_range=(0, 1), num_points=1000, save_file=False, save_path="untitled.csv"):
    
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
        y_scaled = y / max(y)
        points_arr = np.linspace(0, num_points-1, num_points)
        equation_arr = np.array([points_arr, y_scaled])
    except Exception as e:
        raise ValueError(f"Error evaluating expression: {e}")

    # Plot
    plt.plot(equation_arr[0], equation_arr[1])
    plt.title(f"Plot of {equation_str}")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid(True)
    plt.show()

    if save_file == True:
        print(f"Your pulse {equation_str} has been saved to: {save_path}")
        df = pd.DataFrame(equation_arr.T)
        df.to_csv(save_path)

    return equation_arr


#equation="y=sin(x/2pi)^2 + cos(x)"

# Example usage:
#equ2shape(equation, x_range=(0, np.pi), save_file=True)


##### Object Oriented CODE #####
# comment in / out use [cmd + /]


# import numpy as np
# import matplotlib.pyplot as plt
# import pandas as pd
# import re


# class XYFileReader:
#     """Reads and parses XY points from a file."""
    
#     def __init__(self, file_path):
#         self.file_path = file_path
    
#     def read_xy_points(self):
#         with open(self.file_path, 'r', encoding='utf-8') as file:
#             content = file.read()

#         # Extract the XYPOINTS section
#         xy_points_match = re.search(r'##XYPOINTS=(.*?)\n(?:##|$)', content, re.DOTALL)
#         if not xy_points_match:
#             raise ValueError("XYPOINTS section not found in the file.")
        
#         xy_points_str = xy_points_match.group(1).replace('(XY..XY)', '').strip()
#         if not xy_points_str:
#             raise ValueError("XYPOINTS section is empty.")

#         # Convert to list of [x, y]
#         xy_array = [list(map(float, re.split(r',', point.strip()))) 
#                     for point in xy_points_str.split('\n') if point.strip()]
#         return np.array(xy_array, dtype=float)
    
#     def import_file(self):
#         return self.read_xy_points()


# class PulseShape:
#     """Generates pulse shapes from mathematical equations."""
    
#     def __init__(self, equation_str, x_range=(0, 1), num_points=1000):
#         self.equation_str = equation_str
#         self.x_range = x_range
#         self.num_points = num_points
#         self.equation_arr = None
    
#     def _process_equation(self):
#         """Prepare equation string for safe evaluation."""
#         match = re.match(r"y\s*=\s*(.+)", self.equation_str.replace(" ", ""))
#         if not match:
#             raise ValueError("Equation must be in the form 'y=...'")
        
#         expr = match.group(1)
#         expr = re.sub(r'\^', '**', expr)  # Replace ^ with **
#         expr = re.sub(r'(\d)([a-zA-Z])', r'\1*\2', expr)  # 2x -> 2*x
        
#         # Avoid inserting * inside known math functions
#         functions = ['sin', 'cos', 'tan', 'exp', 'log', 'sqrt', 'abs']
#         pattern = rf'\b(?!{"|".join(functions)})([a-zA-Z])\('
#         expr = re.sub(pattern, r'\1*(', expr)
        
#         return expr

#     def generate(self):
#         """Generate the pulse shape based on the equation."""
#         expr = self._process_equation()

#         allowed_names = {
#             'sin': np.sin, 'cos': np.cos, 'tan': np.tan,
#             'exp': np.exp, 'log': np.log, 'sqrt': np.sqrt,
#             'abs': np.abs, 'pi': np.pi, 'e': np.e
#         }

#         x = np.linspace(self.x_range[0], self.x_range[1], self.num_points)
#         allowed_names['x'] = x

#         try:
#             y = eval(expr, {"__builtins__": {}}, allowed_names)
#             y_scaled = y / max(y)
#             points_arr = np.arange(self.num_points)
#             self.equation_arr = np.array([points_arr, y_scaled])
#         except Exception as e:
#             raise ValueError(f"Error evaluating expression: {e}")
        
#         return self.equation_arr
    
#     def plot(self):
#         """Plot the generated pulse shape."""
#         if self.equation_arr is None:
#             raise RuntimeError("No equation data generated. Call generate() first.")
#         plt.plot(self.equation_arr[0], self.equation_arr[1])
#         plt.title(f"Plot of {self.equation_str}")
#         plt.xlabel("x")
#         plt.ylabel("y")
#         plt.grid(True)
#         plt.show()
    
#     def save(self, save_path="untitled.csv"):
#         """Save the pulse shape to a CSV file."""
#         if self.equation_arr is None:
#             raise RuntimeError("No equation data to save. Call generate() first.")
#         df = pd.DataFrame(self.equation_arr.T)
#         df.to_csv(save_path, index=False)
#         print(f"Pulse shape saved to {save_path}")


# # Example usage:
# if __name__ == "__main__":
#     equation = "y=sin(x/2pi)^2 + cos(x)"
#     pulse = PulseShape(equation, x_range=(0, np.pi))
#     pulse.generate()
#     pulse.plot()
#     pulse.save("pulse_shape.csv")
