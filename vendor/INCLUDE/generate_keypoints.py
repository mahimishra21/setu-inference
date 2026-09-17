import os
import json
import multiprocessing
import argparse
import os.path
import cv2
import mediapipe as mp
from tqdm.auto import tqdm
from joblib import Parallel, delayed
import numpy as np
import gc
import warnings

def process_landmarks(landmarks):
    x_list, y_list = [], []
    for landmark in landmarks.landmark:
        x_list.append(landmark.x)
        y_list.append(landmark.y)
    return x_list, y_list


def process_hand_keypoints(results):
    hand1_x, hand1_y, hand2_x, hand2_y = [], [], [], []

    if results.multi_hand_landmarks is not None:
        if len(results.multi_hand_landmarks) > 0:
            hand1 = results.multi_hand_landmarks[0]
            hand1_x, hand1_y = process_landmarks(hand1)

        if len(results.multi_hand_landmarks) > 1:
            hand2 = results.multi_hand_landmarks[1]
            hand2_x, hand2_y = process_landmarks(hand2)

    return hand1_x, hand1_y, hand2_x, hand2_y


def process_pose_keypoints(results):
    # SETU PATCH: mediapipe's Pose() dropped the `upper_body_only` kwarg this
    # repo was written against (removed ~2021). That flag never selected a
    # different model -- it just truncated the standard 33-point BlazePose
    # output to the first 25 (head/torso/arms, excluding hips/legs/feet,
    # matching the [np.nan] * 25 fallback shape used everywhere else in this
    # file). Slicing here reproduces the exact same 25-point output from the
    # modern full-33-point Pose() with no behavior change downstream.
    # SETU PATCH: pre-existing bug, not a version-compat issue. The hand path
    # below already guards multi_hand_landmarks against being None; this
    # function never got the equivalent guard, so any single frame where
    # BlazePose fails to detect a person (poor lighting, briefly out of
    # frame -- exactly what a real counter camera hits) crashed the whole
    # clip with an unhandled AttributeError. The caller already expects and
    # NaN-fills an empty result (see the `pose_x if pose_x else [nan]*25`
    # lines below) -- it just never got the chance to, because this raised
    # first. Returning empty lists here is what the rest of the file was
    # already written to handle.
    pose = results.pose_landmarks
    if pose is None:
        return [], []
    pose_x, pose_y = process_landmarks(pose)
    return pose_x[:25], pose_y[:25]


def swap_hands(left_wrist, right_wrist, hand, input_hand):
    left_wrist_x, left_wrist_y = left_wrist
    right_wrist_x, right_wrist_y = right_wrist
    hand_x, hand_y = hand

    left_dist = (left_wrist_x - hand_x) ** 2 + (left_wrist_y - hand_y) ** 2
    right_dist = (right_wrist_x - hand_x) ** 2 + (right_wrist_y - hand_y) ** 2

    if left_dist < right_dist and input_hand == "h2":
        return True

    if right_dist < left_dist and input_hand == "h1":
        return True

    return False


def process_video(path, save_dir):
    hands = mp.solutions.hands.Hands(
        min_detection_confidence=0.5, min_tracking_confidence=0.5
    )
    # SETU PATCH: upper_body_only removed from mediapipe's Pose() -- see the
    # matching note in process_pose_keypoints(), which restores the same
    # 25-point output by slicing instead.
    pose = mp.solutions.pose.Pose(
        min_detection_confidence=0.5, min_tracking_confidence=0.5
    )

    pose_points_x, pose_points_y = [], []
    hand1_points_x, hand1_points_y = [], []
    hand2_points_x, hand2_points_y = [], []

    # SETU PATCH: path.split("/") silently returns a 1-element list on
    # Windows (backslash paths), so [-2] raised IndexError. os.path is
    # correct on every platform and means the same thing: the immediate
    # parent directory name, which is the gloss this video is labeled with.
    label = os.path.basename(os.path.dirname(path))
    label = "".join([i for i in label if i.isalpha()]).lower()
    uid = os.path.splitext(os.path.basename(path))[0]
    uid = "_".join([label, uid])
    n_frames = 0
    if not os.path.isfile(path):
        warnings.warn(path + " file not found")
    cap = cv2.VideoCapture(path)
    while cap.isOpened():
        ret, image = cap.read()
        if not ret:
            break
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        hand_results = hands.process(image)
        pose_results = pose.process(image)

        hand1_x, hand1_y, hand2_x, hand2_y = process_hand_keypoints(hand_results)
        pose_x, pose_y = process_pose_keypoints(pose_results)

        ## Assign hands to correct positions
        # SETU PATCH: wrist indices 15/16 need pose_x to actually have >16
        # entries. process_pose_keypoints now correctly returns [] when
        # BlazePose detects no pose in a frame (see the patch there), which
        # this swap logic didn't originally account for -- a lone detected
        # hand with no pose in the same frame crashed with IndexError.
        # Skipping the reorder in that case leaves MediaPipe's own hand
        # ordering as-is: not perfect, but not a crash, on a frame the
        # original code could not have handled correctly either way.
        pose_has_wrists = len(pose_x) > 16
        if pose_has_wrists and len(hand1_x) > 0 and len(hand2_x) == 0:
            if swap_hands(
                left_wrist=(pose_x[15], pose_y[15]),
                right_wrist=(pose_x[16], pose_y[16]),
                hand=(hand1_x[0], hand1_y[0]),
                input_hand="h1",
            ):
                hand1_x, hand1_y, hand2_x, hand2_y = hand2_x, hand2_y, hand1_x, hand1_y

        elif pose_has_wrists and len(hand1_x) == 0 and len(hand2_x) > 0:
            if swap_hands(
                left_wrist=(pose_x[15], pose_y[15]),
                right_wrist=(pose_x[16], pose_y[16]),
                hand=(hand2_x[0], hand2_y[0]),
                input_hand="h2",
            ):
                hand1_x, hand1_y, hand2_x, hand2_y = hand2_x, hand2_y, hand1_x, hand1_y

        ## Set to nan so that values can be interpolated in dataloader
        pose_x = pose_x if pose_x else [np.nan] * 25
        pose_y = pose_y if pose_y else [np.nan] * 25

        hand1_x = hand1_x if hand1_x else [np.nan] * 21
        hand1_y = hand1_y if hand1_y else [np.nan] * 21
        hand2_x = hand2_x if hand2_x else [np.nan] * 21
        hand2_y = hand2_y if hand2_y else [np.nan] * 21

        pose_points_x.append(pose_x)
        pose_points_y.append(pose_y)
        hand1_points_x.append(hand1_x)
        hand1_points_y.append(hand1_y)
        hand2_points_x.append(hand2_x)
        hand2_points_y.append(hand2_y)

        n_frames += 1

    cap.release()

    ## Set to nan so that values can be interpolated in dataloader
    pose_points_x = pose_points_x if pose_points_x else [[np.nan] * 25]
    pose_points_y = pose_points_y if pose_points_y else [[np.nan] * 25]

    hand1_points_x = hand1_points_x if hand1_points_x else [[np.nan] * 21]
    hand1_points_y = hand1_points_y if hand1_points_y else [[np.nan] * 21]
    hand2_points_x = hand2_points_x if hand2_points_x else [[np.nan] * 21]
    hand2_points_y = hand2_points_y if hand2_points_y else [[np.nan] * 21]

    save_data = {
        "uid": uid,
        "label": label,
        "pose_x": pose_points_x,
        "pose_y": pose_points_y,
        "hand1_x": hand1_points_x,
        "hand1_y": hand1_points_y,
        "hand2_x": hand2_points_x,
        "hand2_y": hand2_points_y,
        "n_frames": n_frames,
    }
    with open(os.path.join(save_dir, f"{uid}.json"), "w") as f:
        json.dump(save_data, f)

    hands.close()
    pose.close()
    del hands, pose, save_data
    gc.collect()


def load_file(path, include_dir):
    with open(path, "r") as fp:
        data = fp.read()
        data = data.split("\n")
    data = list(map(lambda x: os.path.join(include_dir, x), data))
    return data


def load_train_test_val_paths(args):
    train_paths = load_file(
        f"train_test_paths/{args.dataset}_train.txt", args.include_dir
    )
    val_paths = load_file(f"train_test_paths/{args.dataset}_val.txt", args.include_dir)
    test_paths = load_file(
        f"train_test_paths/{args.dataset}_test.txt", args.include_dir
    )
    return train_paths, val_paths, test_paths


def save_keypoints(dataset, file_paths, mode):
    save_dir = os.path.join(args.save_dir, f"{dataset}_{mode}_keypoints")
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    Parallel(n_jobs=n_cores, backend="multiprocessing")(
        delayed(process_video)(path, save_dir)
        for path in tqdm(file_paths, desc=f"processing {mode} videos")
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate keypoints from Mediapipe")
    parser.add_argument(
        "--include_dir",
        default="",
        type=str,
        required=True,
        help="path to the location of INCLUDE/INCLUDE50 videos",
    )
    parser.add_argument(
        "--save_dir",
        default="",
        type=str,
        required=True,
        help="location to output json file",
    )
    parser.add_argument(
        "--dataset", default="include", type=str, help="options: include or include50"
    )
    args = parser.parse_args()

    n_cores = multiprocessing.cpu_count()
    train_paths, val_paths, test_paths = load_train_test_val_paths(args)

    save_keypoints(args.dataset, val_paths, "val")
    save_keypoints(args.dataset, test_paths, "test")
    save_keypoints(args.dataset, train_paths, "train")
