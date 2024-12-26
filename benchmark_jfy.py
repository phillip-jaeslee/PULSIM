import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

"""
file_path = 'Hypesec_1000_60.xlsx'
#file_path = 'hard20_100.xlsx'
#file_path = 'sin600_hard20_sin600_jsl.xlsx'
#file_path = 'GaQ5_3m_20u_GaQ5rev_3m.xlsx'
#file_path = 'hard20_100.xlsx'
#file_path = 'sine600.xlsx'
data = pd.read_excel(file_path, skiprows=5)
data[';Offset[Hz]'] = data[';Offset[Hz]'].str.replace(':', '', regex=False)

data = data.astype(float)


axs[3].plot(data[';Offset[Hz]'], data['Mz'], label="Mz", color='red', alpha=1)
axs[3].plot(data[';Offset[Hz]'], data['My'], label="My", color='blue')
axs[3].plot(data[';Offset[Hz]'], data['Mx'], label="Mx", color='green')
axs[3].set_title('Excitation Profile Calculated by Topspin', fontsize=title_font_size, fontname=font_name)
axs[3].set_xlabel('frequency (Hz)', fontsize=label_font_size, fontname=font_name)
axs[3].set_ylabel('flip', fontsize=label_font_size, fontname=font_name)
axs[3].tick_params(axis='both', labelsize=ticks_font_size)
axs[3].legend(loc="upper right")


fig.savefig('arial_benchmark_Hypsec_adiabatic.svg', dpi=600)
#axs[1].legend(loc="lower right", bbox_to_anchor=(1.1, 0))
plt.show()
"""

import os
import pandas as pd
import matplotlib.pyplot as plt

# Define the folder path containing the Excel files
folder_path = 'Excitation_profiles/'  # Replace with the path to your folder containing Excel files

# Get a list of all Excel files in the folder
excel_files = [f for f in os.listdir(folder_path) if f.endswith('.xlsx')]

# Initialize the plot
plt.figure(figsize=(8, 4))

# Loop through each Excel file and plot the data
for file in excel_files:
    file_path = os.path.join(folder_path, file)
    
    # Read the Excel file
    data = pd.read_excel(file_path, skiprows=5)
    
    # Clean and convert data
    data[';Offset[Hz]'] = data[';Offset[Hz]'].str.replace(':', '', regex=False)
    data = data.astype(float)
    
    # Extract the file name without the extension
    base_name = os.path.splitext(file)[0]
    
    # Plot data with labels using the base file name
    plt.plot(data[';Offset[Hz]'], data['Mz'], label=f'{base_name} - Mz', alpha=0.8)
    #plt.plot(data[';Offset[Hz]'], data['My'], label=f'{base_name} - My', alpha=0.8)
    #plt.plot(data[';Offset[Hz]'], data['Mx'], label=f'{base_name} - Mx', alpha=0.8)

# Customize the plot
plt.title('Excitation Profile', fontsize=18, fontname='Arial')
plt.xlabel('frequency (Hz)', fontsize=12, fontname='Arial')
plt.ylim(top=1.1, bottom=-1.1)
plt.ylabel('flip', fontsize=12, fontname='Arial')
plt.tick_params(axis='both', labelsize=10)
plt.legend(loc="upper right")

# Save the figure
plt.savefig('combined_plot.svg', dpi=600)

# Display the plot
plt.show()
