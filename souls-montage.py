#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import glob
from enum import Enum

import numpy as np
import cv2

from tqdm import tqdm

from game_config import *

RESIZE_FACTOR = 0.6
MIN_MS_BETWEEN_ATTEMPTS = 5 * 1000  # 5 seconds

game_config = GAME_CONFIG["bloodborne"]

vfile = sorted(glob.glob("devclips/*.mp4"))[0]

cap = cv2.VideoCapture(vfile)

frames_per_sec = int(cap.get(cv2.CAP_PROP_FPS))
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

tmpl = cv2.imread("templates/bloodborne/ludwig_the_accursed.png", cv2.IMREAD_GRAYSCALE)
tmpl = cv2.resize(tmpl, (0, 0), fx=RESIZE_FACTOR, fy=RESIZE_FACTOR)

you_died_tmpl = cv2.imread("templates/bloodborne/you_died.png", cv2.IMREAD_GRAYSCALE)
you_died_tmpl = cv2.resize(you_died_tmpl, (0, 0), fx=RESIZE_FACTOR, fy=RESIZE_FACTOR)


class FrameData:
    def __init__(
        self, frame_idx, boss_active=False, boss_hp_pct=-1, you_died_visible=False
    ):
        self.frame_idx = frame_idx
        self.boss_active = boss_active
        self.boss_hp_pct = boss_hp_pct
        self.you_died_visible = you_died_visible

    def __repr__(self):
        return "-{}--{}-{}--{}--".format(
            str(self.frame_idx).center(8, "-"),
            "BOSS" if self.boss_active else "----",
            str(self.boss_hp_pct).rjust(4, "-") if self.boss_active else "----",
            "DIED" if self.you_died_visible else "----",
        )


frame_data: dict[int, FrameData] = {}


def frame_to_ms(frame_idx: int):
    return int(frame_idx * (1 / frames_per_sec) * 1000)


def ms_to_frames(ms: int):
    return int(ms * (frames_per_sec / 1000))


def get_box(frame, which: str):
    height, width = frame.shape[0], frame.shape[1]

    y1 = int(height * game_config[f"{which}_y_start_pct"])
    y2 = int(height * game_config[f"{which}_y_end_pct"])
    x1 = int(width * game_config[f"{which}_x_start_pct"])
    x2 = int(width * game_config[f"{which}_x_end_pct"])

    return frame[y1:y2, x1:x2]


def process_frame(frame, frame_idx) -> FrameData:
    frame_data = FrameData(frame_idx)

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = cv2.resize(frame, (0, 0), fx=RESIZE_FACTOR, fy=RESIZE_FACTOR)

    you_died_box = get_box(frame, "you_died")
    match = cv2.matchTemplate(you_died_box, you_died_tmpl, cv2.TM_CCOEFF_NORMED)
    if np.amax(match) > 0.5:
        frame_data.you_died_visible = True

    boss_bar_box = get_box(frame, "boss_bar")
    match = cv2.matchTemplate(boss_bar_box, tmpl, cv2.TM_CCORR_NORMED)
    if np.amax(match) > 0.75:
        frame_data.boss_active = True

    return frame_data


def print_results():
    class State(Enum):
        NO_BOSS = 0
        ATTEMPT_STARTED = 1

    state = State.NO_BOSS
    attempts = []
    last_attempt_start = -1
    frames_without_boss = 0

    for frame in tqdm(sorted(frame_data.keys()), desc="Processing frame data"):
        if frame_data[frame].boss_active:
            frames_without_boss = 0
        else:
            frames_without_boss += 1

        if state == State.NO_BOSS and frame_data[frame].boss_active:
            if len(attempts) > 0:
                time_from_last_attempt = frame_to_ms(frame) - frame_to_ms(attempts[-1][1])
            else:
                # when there were no attempts yet then we don't need a cooldown
                time_from_last_attempt = MIN_MS_BETWEEN_ATTEMPTS

            if time_from_last_attempt >= MIN_MS_BETWEEN_ATTEMPTS:
                last_attempt_start = frame
                state = State.ATTEMPT_STARTED

        if state == State.ATTEMPT_STARTED and frame_data[frame].you_died_visible:
            end = frame
            attempts.append((last_attempt_start, end))
            state = State.NO_BOSS

    print()
    for idx, attempt in enumerate(attempts):
        start = frame_to_ms(attempt[0])
        end = frame_to_ms(attempt[1])
        length = (end - start) / 1000

        print(f"Attempt #{idx + 1}: took {length}ms ({start}-{end}ms)")


with tqdm(desc=os.path.basename(vfile), total=frame_count) as progress:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        frame_data[frame_idx] = process_frame(frame, frame_idx)

        progress.update()

print_results()

cap.release()
