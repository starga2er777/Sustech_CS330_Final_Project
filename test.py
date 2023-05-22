import os
import imagej

# ij = imagej.init()
ij = imagej.init('./Fiji.app', mode=imagej.Mode.HEADLESS)

# ij.py.show(image, cmap='gray')

script_path = '../segment.bsh'
with open(script_path, 'r') as raw_script:
    script = raw_script.read()


# set args
input_directory = '../test_input/'
output_directory = '../test_output/'
model_path = './c3v3.model'

args = {"input_dir": input_directory,
        "output_dir": output_directory,
        "model_path": model_path,
        "guassian_sigma": 5.0}

# run segmentation script
# TODO
result = ij.py.run_script("BeanShell", script, args)

print("-------- Segmentation Complete --------")
print("Start Analysis")

# apply analysis for the segmented images
script_path = '../analysis.bsh'
with open(script_path, 'r') as raw_script:
    script = raw_script.read()

args = {"input_dir": output_directory}

result = ij.py.run_script("BeanShell", script, args)

file_names = ij.py.from_java(result.getOutput("file_names"))
area_size = ij.py.from_java(result.getOutput("answer"))
print(file_names)
print(area_size)

print("-------- Analysis Complete --------")

end = 1