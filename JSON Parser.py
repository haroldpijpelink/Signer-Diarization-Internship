import json
import os
import pandas as pd

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("json", help="path to folder with all json files")
args = parser.parse_args()
path_to_json = args.json
#path_to_json = "D:/OpenPose Data Signer Diarization/Rekeshort/"


def parse_json_body(json_list):

    number_of_keyps = len(json_list)/3
    if number_of_keyps != 25:
        return "Body Parts Missing"

    # Turn the JSON output into a dictionary per body part
    keypoint_list = ["Nose", "Neck", "RShoulder", "RElbow", "RWrist", "LShoulder", "LElbow", "LWrist", "MidHip", "RHip", "RKnee", "RAnkle",
                     "LHip", "LKnee", "LAnkle", "REye", "LEye", "REar", "LEar", "LBigToe", "LSmallToe", "LHeel", "RBigToe", "RSmallToe", "RHeel", "Background"]
    bodyparts = {}
    for number in range(1, 26):
        scores = json_list[(number-1)*3: number*3]
        bodyparts[keypoint_list[number-1]] = scores

    present_bodyparts = bodyparts.copy()
    for keypoint in keypoint_list[:25]:
        if present_bodyparts.get(keypoint)[2] == 0:
            present_bodyparts.pop(keypoint)
    bodypartslist = []
    for key in present_bodyparts:
        bodypartslist.append(
            (present_bodyparts[key][0], present_bodyparts[key][1]))

    df_body = pd.DataFrame([bodypartslist], columns=present_bodyparts.keys())

    return df_body


def parse_json_hand(json_list, direction):

    number_of_keyps = len(json_list)/3
    if number_of_keyps != 21:
        return "Fingers Missing"

    # Turn the JSON output into a dictionary per body part
    if (direction == "left") | (direction == "Left"):
        keypoint_list = ["LThumb1CMC", "LThumb2Knuckles", "LThumb3IP", "LThumb4FingerTip", "LIndex1Knuckles", "LIndex2PIP", "LIndex3DIP", "LIndex4FingerTip", "LMiddle1Knuckles",
                         "LMiddle2PIP", "LMiddle3DIP", "LMiddle4FingerTip", "LRing1Knuckles", "LRing2PIP", "LRing3DIP", "LRing4FingerTip", "LPinky1Knuckles", "LPinky2PIP", "LPinky3DIP", "LPinky4FingerTip"]

    if (direction == "right") | (direction == "Right"):
        keypoint_list = ["RThumb1CMC", "RThumb2Knuckles", "RThumb3IP", "RThumb4FingerTip", "RIndex1Knuckles", "RIndex2PIP", "RIndex3DIP", "RIndex4FingerTip", "RMiddle1Knuckles",
                         "RMiddle2PIP", "RMiddle3DIP", "RMiddle4FingerTip", "RRing1Knuckles", "RRing2PIP", "RRing3DIP", "RRing4FingerTip", "RPinky1Knuckles", "RPinky2PIP", "RPinky3DIP", "RPinky4FingerTip"]

    bodyparts = {}
    for number in range(1, len(keypoint_list)+1):
        scores = json_list[(number-1)*3: number*3]
        bodyparts[keypoint_list[number-1]] = scores

    present_bodyparts = bodyparts.copy()
    # Background is last, don't need that stuff
    for keypoint in keypoint_list[:21]:
        if present_bodyparts.get(keypoint)[2] == 0:
            present_bodyparts.pop(keypoint)
    bodypartslist = []
    for key in present_bodyparts:
        bodypartslist.append(
            (present_bodyparts[key][0], present_bodyparts[key][1]))

    df_left_hand = pd.DataFrame(
        [bodypartslist], columns=present_bodyparts.keys())

    return df_left_hand


keypoint_list_body = ["Nose", "Neck", "RShoulder", "RElbow", "RWrist", "LShoulder", "LElbow", "LWrist", "MidHip", "RHip", "RKnee", "RAnkle",
                      "LHip", "LKnee", "LAnkle", "REye", "LEye", "REar", "LEar", "LBigToe", "LSmallToe", "LHeel", "RBigToe", "RSmallToe", "RHeel", "Background"]
keypoint_list_left_hand = ["LThumb1CMC", "LThumb2Knuckles", "LThumb3IP", "LThumb4FingerTip", "LIndex1Knuckles", "LIndex2PIP", "LIndex3DIP", "LIndex4FingerTip", "LMiddle1Knuckles",
                           "LMiddle2PIP", "LMiddle3DIP", "LMiddle4FingerTip", "LRing1Knuckles", "LRing2PIP", "LRing3DIP", "LRing4FingerTip", "LPinky1Knuckles", "LPinky2PIP", "LPinky3DIP", "LPinky4FingerTip"]
keypoint_list_right_hand = ["RThumb1CMC", "RThumb2Knuckles", "RThumb3IP", "RThumb4FingerTip", "RIndex1Knuckles", "RIndex2PIP", "RIndex3DIP", "RIndex4FingerTip", "RMiddle1Knuckles",
                            "RMiddle2PIP", "RMiddle3DIP", "RMiddle4FingerTip", "RRing1Knuckles", "RRing2PIP", "RRing3DIP", "RRing4FingerTip", "RPinky1Knuckles", "RPinky2PIP", "RPinky3DIP", "RPinky4FingerTip"]

json_files = [pos_json for pos_json in os.listdir(
    path_to_json) if pos_json.endswith('.json')]
print('Found: ', len(json_files), 'json keypoint frame files')


left_dataframe_body = pd.DataFrame(columns=keypoint_list_body)
left_dataframe_LH = pd.DataFrame(columns=keypoint_list_left_hand)
left_dataframe_RH = pd.DataFrame(columns=keypoint_list_right_hand)

right_dataframe_body = pd.DataFrame(columns=keypoint_list_body)
right_dataframe_LH = pd.DataFrame(columns=keypoint_list_left_hand)
right_dataframe_RH = pd.DataFrame(columns=keypoint_list_right_hand)

for index, file in enumerate(json_files):
    temp_df = json.load(open(os.path.join(path_to_json, file)))
    if len(temp_df['people']) > 0:
      leftlist = temp_df['people'][0]["pose_keypoints_2d"]
      left_left_hand_list = temp_df['people'][0]["hand_left_keypoints_2d"]
      left_right_hand_list = temp_df['people'][0]["hand_right_keypoints_2d"]

      left_dataframe_body = left_dataframe_body.append(
          pd.DataFrame(parse_json_body(leftlist)))
      left_dataframe_LH = left_dataframe_LH.append(pd.DataFrame(
          parse_json_hand(left_left_hand_list, direction="left")))
      left_dataframe_RH = left_dataframe_RH.append(pd.DataFrame(
          parse_json_hand(left_right_hand_list, direction="right")))
    if len(temp_df['people']) > 1:
      rightlist = temp_df['people'][1]["pose_keypoints_2d"]
      right_left_hand_list = temp_df['people'][1]["hand_left_keypoints_2d"]
      right_right_hand_list = temp_df['people'][1]["hand_right_keypoints_2d"]

      right_dataframe_body = right_dataframe_body.append(
          pd.DataFrame(parse_json_body(rightlist)))

      right_dataframe_LH = right_dataframe_LH.append(pd.DataFrame(
          parse_json_hand(right_left_hand_list, direction="left")))
      right_dataframe_RH = right_dataframe_RH.append(pd.DataFrame(
          parse_json_hand(right_right_hand_list, direction="right")))

left_final = pd.concat(
    [left_dataframe_body, left_dataframe_LH, left_dataframe_RH], axis=1)
#print(left_final.head())

right_final = pd.concat(
    [right_dataframe_body, right_dataframe_LH, right_dataframe_RH], axis=1)

# Save dataframes as csv
left_final.to_csv(path_or_buf=path_to_json +
                  r"\\left_dataframe.csv", index=False)
right_final.to_csv(path_or_buf=path_to_json +
                   r"\\right_dataframe.csv", index=False)
print(f"Writing Done")
