import os
import imagej
import numpy as np
import pandas as pd
from pandas import DataFrame
import matplotlib.pyplot as plt

# TODO: scales taken as inputs
scale_len = 400 # pixels
scale_size = 100 # μm

# μm per pixel
unit = scale_size / scale_len

# ij = imagej.init()
ij = imagej.init('./Fiji.app', mode=imagej.Mode.HEADLESS)

# ij.py.show(image, cmap='gray')

script_path = '../segment.bsh'
with open(script_path, 'r') as raw_script:
    script = raw_script.read()


# set args
input_directory = '../raw_input/'
segment_directory = '../segment_output/'
analysis_directory = '../threshold_output/'
model_path = './c3v3.model'

args = {"input_dir": input_directory,
        "output_dir": segment_directory,
        "model_path": model_path}

# run segmentation script - takes quite a long time
# result = ij.py.run_script("BeanShell", script, args)

print("-------- Segmentation Complete --------")
print("Start Analysis")

# apply analysis for the segmented images
script_path = '../analysis.bsh'
with open(script_path, 'r') as raw_script:
    script = raw_script.read()

# process the image, then calculate the data
args = {"input_dir": segment_directory,
        "output_dir": analysis_directory,
        "noise_rad": 8.0}

result = ij.py.run_script("BeanShell", script, args)

file_names = list(ij.py.from_java(result.getOutput("file_names")))
elem_percentage = list(ij.py.from_java(result.getOutput("answer")))
elem_area_lists = list(ij.py.from_java(result.getOutput("particles")))

# convert from java list to python list
new_list = []

for java_list in elem_percentage:
    py_list = list(java_list)
    new_list.append([area_val * unit for area_val in py_list])

elem_percentage = new_list

new_list = []

for java_list in elem_area_lists:
    py_list = list(java_list)
    new_list.append([area_val * unit for area_val in py_list])

elem_area_lists = new_list


# OUTPUTS:
# file_names - file name according to the input images
# elem_percentage - area size of each element(gravel particles, clay and cavity)
# elem_area_lists - lists of size of each gravel particle



for i in range(len(elem_area_lists)):
    area_list = elem_area_lists[i]
    figure, axes = plt.subplots(1, 4, figsize=(20, 5))
    data_range_500 = [value for value in area_list if 0 <= value <= 10000]
    axes[0].hist(data_range_500, bins=20, alpha=0.5)
    axes[0].set_title('Distribution of Area for Small Particles')
    axes[0].set_xlabel('Area')
    axes[0].set_ylabel('Frequency')

    data_range_10000 = [value for value in area_list if value > 10000]
    axes[1].hist(data_range_10000, bins=20, alpha=0.5)
    axes[1].set_title('Distribution of Area for Big Particles')
    axes[1].set_xlabel('Area')
    axes[1].set_ylabel('Frequency')

    # Plot boxplot of area
    axes[2].boxplot(area_list)
    axes[2].set_title('Boxplot of Area for Particles')
    axes[2].set_xlabel('Area')
    axes[2].set_ylabel('Value')


    # Create a pie chart
    labels = ['Particles', 'Clay', 'Cavity']
    axes[3].pie(elem_percentage[i], labels=labels, autopct='%1.1f%%')
    axes[3].set_title('Distribution of Categories')

    plt.tight_layout()
    plt.savefig('../graph_output/' + file_names[i])


print("-------- Processing Complete --------")

