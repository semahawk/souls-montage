#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import glob
import concurrent.futures
from enum import Enum

import numpy as np
import cv2

from tqdm import tqdm

from game_config import *

THREADS_NUM = 3
RESIZE_FACTOR = 0.6

config = GAME_CONFIG["bloodborne"]

vfile = glob.glob("devclips/*.mp4")[0]

cap = cv2.VideoCapture(vfile)

frames_per_sec = int(cap.get(cv2.CAP_PROP_FPS))
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

tmpl = cv2.imread("templates/bloodborne/ludwig_the_accursed.png", cv2.IMREAD_GRAYSCALE)
tmpl = cv2.resize(tmpl, (0, 0), fx=RESIZE_FACTOR, fy=RESIZE_FACTOR)

# [frame index] = True/False if the boss is active in that frame
boss_in_frame = {}


def frame_to_ms(frame_idx):
    return int(frame_idx * (1 / frames_per_sec) * 1000)


def getsubframe(frame, x1, y1, x2, y2):
    return np.asarray([row[x1:x2] for row in frame[y1:y2]])


def process_frame(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = cv2.resize(frame, (0, 0), fx=RESIZE_FACTOR, fy=RESIZE_FACTOR)

    height, width = frame.shape

    boss_bar_top = int(height * config["boss_bar_y_start_pct"])
    boss_bar_bottom = int(height * config["boss_bar_y_end_pct"])
    boss_bar_left = int(width * config["boss_bar_x_start_pct"])
    boss_bar_right = int(width * config["boss_bar_x_end_pct"])
    boss_bar = getsubframe(frame,
        boss_bar_left,
        boss_bar_top,
        boss_bar_right - int((boss_bar_right - boss_bar_left) / 2),
        boss_bar_bottom - int((boss_bar_bottom - boss_bar_top) / 2)
    )

    match = cv2.matchTemplate(boss_bar, tmpl, cv2.TM_CCORR_NORMED)

    threshold = 0.85
    if np.amax(match) > threshold:
        return True

    return False


def print_results():
    class State(Enum):
        NO_BOSS = 0
        BOSS_ACTIVE = 1

    attempts = []
    last_attempt_start = -1
    state = State.NO_BOSS

    for frame in tqdm(sorted(boss_in_frame.keys()), desc="Processing frame data"):
        if state == State.NO_BOSS and boss_in_frame[frame] == True:
            last_attempt_start = frame_to_ms(frame)
            state = State.BOSS_ACTIVE
        if state == State.BOSS_ACTIVE and boss_in_frame[frame] == False:
            end = frame_to_ms(frame)
            attempts.append((last_attempt_start, end))
            state = State.NO_BOSS

    print()
    for idx, attempt in enumerate(attempts):
        start = attempt[0]
        end = attempt[1]
        length = (end - start) / 1000

        print(f"Attempt #{idx + 1}: took {length}ms ({start}-{end}ms)")


with tqdm(desc=os.path.basename(vfile), total=frame_count) as progress:
    should_end = False

    while cap.isOpened():
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for i in range(THREADS_NUM):
                ret, frame = cap.read()
                if not ret:
                    should_end = True
                    break

                frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

                future = executor.submit(process_frame, frame)
                future.add_done_callback(lambda _: progress.update())

                boss_present = future.result()
                boss_in_frame[frame_idx] = boss_present

        if should_end:
            break

print_results()

cap.release()
