#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import glob
import concurrent.futures
from enum import Enum

import numpy as np
import cv2

from tqdm import tqdm

THREADS_NUM = 3
RESIZE_FACTOR = 0.5

vfile = glob.glob("devclips/*.mp4")[0]

cap = cv2.VideoCapture(vfile)

frames_per_sec = int(cap.get(cv2.CAP_PROP_FPS))
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

tmpl = cv2.imread("templates/bloodborne/ludwig_the_accursed.png", cv2.IMREAD_GRAYSCALE)
tmpl_h, tmpl_w = tmpl.shape
tmpl = cv2.resize(tmpl, (0, 0), fx=RESIZE_FACTOR, fy=RESIZE_FACTOR)

# [frame index] = True/False if the boss is active in that frame
boss_in_frame = {}


def frame_to_ms(frame_idx):
    return int(frame_idx * (1 / frames_per_sec) * 1000)


def process_frame(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = cv2.resize(frame, (0, 0), fx=RESIZE_FACTOR, fy=RESIZE_FACTOR)

    match = cv2.matchTemplate(frame, tmpl, cv2.TM_CCOEFF_NORMED)

    threshold = 0.90
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
        print(f"Attempt #{idx + 1}: {attempt[0]}ms - {attempt[1]}ms")


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
