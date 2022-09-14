import math
import pympi
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("json", help="path to folder with all json files")
parser.add_argument(
    "left_dataframe", help="path to folder with left_dataframe.csv")
parser.add_argument("right_dataframe",
                    help="path to folder with right_dataframe.csv")
parser.add_argument("elan", help="path to elan file")
parser.add_argument("center", help="boolean for centering around the nose keypoints")
parser.add_argument("out", help="output file location")
args = parser.parse_args()
if args.center == "True":
  print("Centering x keypoints")
path_to_json = args.json

input_file_left = args.left_dataframe
left_df = pd.read_csv(input_file_left)
input_file_right = args.right_dataframe
right_df = pd.read_csv(input_file_right)
elan_file_path = args.elan
out = args.out
# Not all rows are assigned the correct ID in OpenPose
# Find affected rows using the x-coordinate of the nose
# append all to an array and find the closest two x coordinates
# Then use those as thresholds for exchanging rows with the right dataframe
xcoordlist = []
#print(left_df.columns)
for coordinate in left_df["Nose"]:
    xcoordlist.append(float(coordinate.split(', ')[0].strip("(")))
plt.hist(xcoordlist)
plt.savefig(os.path.join(path_to_json, "nosecoords.png"))

# Histogram shows two clusters: one around 225 and one around 450.
# Let's use the middle as decision boundary, so around everything larger than 350 is affected.

affected_list = [True if x > 350 else False for x in xcoordlist]

temp = left_df.copy()

for index, row in enumerate(affected_list):
    if row:
        left_df.iloc[index] = right_df.iloc[index]
        right_df.iloc[index] = temp.iloc[index]

print("IDs corrected")

# Add labels
# https://pypi.org/project/pympi-ling/
# replace with path to elan file
eaf = pympi.Elan.Eaf(elan_file_path)
ts = eaf.timeslots

# timestampts for person 1 and person 2
t1 = eaf.tiers.get("Translation Engl 1")[0]
t2 = eaf.tiers.get("Translation Engl 2")[0]

# frames in which person 1 signed
t1_frames = []
for key, value in t1.items():
    for frame in range(math.floor(ts[value[0]]*0.025), math.ceil(ts[value[1]]*0.025)):
        t1_frames.append(frame)

# frames in which person 2 signed
t2_frames = []
for key, value in t2.items():
    for frame in range(math.floor(ts[value[0]]*0.025), math.ceil(ts[value[1]]*0.025)):
        t2_frames.append(frame)

labels_left = []
labels_right = []
#print(math.ceil(ts[value[1]]*0.025))
# replace with number of frames
frames = left_df.shape[0]

print("Frames: ", str(frames))
#frames = 36515
#frames = 26989
'''
for i in range(frames):
    if (i in t1_frames and i in t2_frames):
        labels.append(3)
        continue
    if (i in t1_frames and i not in t2_frames):
        labels.append(1)
        continue
    if (i not in t1_frames and i in t2_frames):
        labels.append(2)
        continue
    if (i not in t1_frames and i not in t2_frames):
        labels.append(0)
        continue
'''
for i in range(frames):
    if (i in t1_frames):
        labels_left.append(1)
        continue
    else:
        labels_left.append(0)
        continue

for i in range(frames):
    if (i in t2_frames):
        labels_right.append(1)
        continue
    else:
        labels_right.append(0)
        continue

print("Labels Added")

# Remove background column
left_df = left_df.drop(columns=['Background'])
right_df = right_df.drop(columns=['Background'])

#Backfill all missing data
left_df = left_df.fillna(method="pad")
right_df = right_df.fillna(method="pad")

print("Missing Data Filled")

# Turn tuples into two separate columns
nose_left = []

for index, row in enumerate(left_df["Nose"].tolist()):
  
  if isinstance(row, str):

    nose_left.append(float(row.strip("(").strip(")").split(", ")[0]))

nose_right = []
for index, row in enumerate(right_df["Nose"].tolist()):
  if isinstance(row, str):

    nose_right.append(float(row.strip("(").strip(")").split(", ")[0]))

print("X centering keypoints extracted")


def split_tuples(dataframe, column, left=True):
    string_x = str(column) + "_x"
    string_y = str(column) + "_y"
    x_list = []
    y_list = []

    for index, row in enumerate(dataframe[column].tolist()):
        if isinstance(row, str):
            tuple_list = row.strip("(").strip(")").split(", ")
            if args.center == "True":
              if left == True:
                x_list.append(float(tuple_list[0])-float(nose_left[index]))
              if left == False:
                x_list.append(float(tuple_list[0])-float(nose_right[index]))
            else:
              x_list.append(float(tuple_list[0]))
            y_list.append(float(tuple_list[1]))
        else:
            x_list.append(0)
            y_list.append(0)
    dataframe[string_x] = x_list
    dataframe[string_y] = y_list
    dataframe = dataframe.drop(column, axis=1)
    return dataframe



for column in left_df.columns:
    left_df = split_tuples(left_df, column, left=True)

for column in right_df.columns:
    right_df = split_tuples(right_df, column, left=False)
print("Keypoints Centered")
print("Tuples Split")

def mass_rename(dataframe, name):
    name_dict = {}
    cols = dataframe.columns
    for column in cols:
        name_dict[column] = column + name
    dataframe.rename(columns=name_dict, inplace=True)
    return dataframe


left_df = mass_rename(left_df, "_left_person")
right_df = mass_rename(right_df, "_right_person")

print("Columns Renamed")

full_df = pd.concat([left_df, right_df], axis=1)
print("Left", labels_left)
print("Right", labels_right)
full_df['left label'] = labels_left
full_df['right label'] = labels_right
full_df = full_df.drop(full_df.columns[0], axis = 1)
out_path = os.path.join(out, "final_dataframe_clean.csv")
print(out_path)
full_df.to_csv(path_or_buf=out_path, index=True)
print(full_df.shape)
print("Preprocessing Done")
