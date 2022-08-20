#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import glob
import math
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

frames_per_sec = cap.get(cv2.CAP_PROP_FPS)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

tmpl = cv2.imread("templates/bloodborne/ludwig_the_accursed.png", cv2.IMREAD_UNCHANGED)
tmpl = cv2.resize(tmpl, (0, 0), fx=RESIZE_FACTOR, fy=RESIZE_FACTOR)

you_died_tmpl = cv2.imread("templates/bloodborne/you_died.png", cv2.IMREAD_UNCHANGED)
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
        return "--{}-{}--{}-{}--{}--".format(
            ms_to_hms(frame_to_ms(self.frame_idx)),
            str(self.frame_idx).ljust(8, "-"),
            "BOSS" if self.boss_active else "----",
            str(int(self.boss_hp_pct)).rjust(3, "-")
            if self.boss_hp_pct != -1
            else "---",
            "DIED" if self.you_died_visible else "----",
        )


frame_data: dict[int, FrameData] = {}


def hex_to_cv2_color(hex_: int):
    r = (hex_ & 0xFF0000) >> 16
    g = (hex_ & 0x00FF00) >> 8
    b = (hex_ & 0x0000FF) >> 0
    return np.array([b, g, r])


def frame_to_ms(frame_idx: int):
    return int(frame_idx * (1 / frames_per_sec) * 1000)


def ms_to_frames(ms: int):
    return int(ms * (frames_per_sec / 1000))


def ms_to_hms(ms: int):
    s = math.floor(ms / 1000)
    m = math.floor(s / 60)
    h = math.floor(m / 60)

    return "{:02d}:{:02d}:{:02d}".format(h, m % 60, s % 60)


def get_box(frame, which: str):
    height, width = frame.shape[0], frame.shape[1]

    y1 = int(height * game_config[f"{which}_y_start_pct"])
    y2 = int(height * game_config[f"{which}_y_end_pct"])
    x1 = int(width * game_config[f"{which}_x_start_pct"])
    x2 = int(width * game_config[f"{which}_x_end_pct"])

    return frame[y1:y2, x1:x2]


def process_frame(orig_frame, frame_idx) -> FrameData:
    frame_data = FrameData(frame_idx)

    frame = cv2.cvtColor(orig_frame, cv2.COLOR_BGR2GRAY)
    frame = cv2.resize(frame, (0, 0), fx=RESIZE_FACTOR, fy=RESIZE_FACTOR)
    _boss_name_tmpl = cv2.cvtColor(tmpl, cv2.COLOR_BGR2GRAY)
    _you_died_tmpl = cv2.cvtColor(you_died_tmpl, cv2.COLOR_BGR2GRAY)

    you_died_box = get_box(frame, "you_died")
    match = cv2.matchTemplate(you_died_box, _you_died_tmpl, cv2.TM_CCOEFF_NORMED)
    if np.amax(match) > 0.5:
        frame_data.you_died_visible = True

    # try to find the boss' name
    boss_bar_box = get_box(frame, "boss_bar")
    match = cv2.matchTemplate(boss_bar_box, _boss_name_tmpl, cv2.TM_CCORR_NORMED)
    boss_active_confidence = np.amax(match)

    # and try to see if the boss' hp bar is active
    # by checking the dominant color of this area and checking if in range
    boss_hp_bar_box = get_box(orig_frame, "boss_hp_bar")
    boss_hp_bar_data = np.reshape(boss_hp_bar_box, (-1, 3))
    boss_hp_bar_data = np.float32(boss_hp_bar_data)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    flags = cv2.KMEANS_RANDOM_CENTERS
    _, _, centers = cv2.kmeans(boss_hp_bar_data, 1, None, criteria, 10, flags)

    img = np.ones((1, 1, 3), dtype=np.uint8)
    img[:, :] = centers[0]

    lowerb = hex_to_cv2_color(game_config["boss_bar_dominant_color_lower_bound"])
    upperb = hex_to_cv2_color(game_config["boss_bar_dominant_color_upper_bound"])
    mask = cv2.inRange(img, lowerb, upperb)

    # bump the confidence by a fair amount if the dominant color of the boss' hp
    # box is what we would expect when the boss is active
    if mask[0][0] == 255:
        boss_active_confidence += 0.5

    if boss_active_confidence > 0.85:
        frame_data.boss_active = True

    # calculate how much hp the boss has (if decently confident that boss is active)
    if boss_active_confidence > 0.5:
        # when YOU DIED is visible then the screen has a semi-transparent overlay
        # making the white more gray
        if frame_data.you_died_visible:
            white_lowerb, white_upperb = (115, 115, 115), (125, 125, 125)
        else:
            white_lowerb, white_upperb = (220, 220, 220), (255, 255, 255)

        # extract the white/gray tip(s)
        whites = cv2.inRange(boss_hp_bar_box, white_lowerb, white_upperb)
        h, w = boss_hp_bar_box.shape[0], boss_hp_bar_box.shape[1]

        # go from the right towards the left side of the bar, and check if we
        # encounter any white pixels
        for x in reversed(range(1, w)):
            if np.count_nonzero(whites[:, x - 1 : x] > 0):
                frame_data.boss_hp_pct = x / w * 100
                break

    return frame_data


def print_results():
    class State(Enum):
        NO_BOSS = 0
        ATTEMPT_STARTED = 1

    state = State.NO_BOSS
    attempts = []
    last_attempt_start = -1
    frames_without_boss = 0

    logfile = open('.frame_data.log', 'w')

    for frame_idx in tqdm(sorted(frame_data.keys()), desc="Processing frame data"):
        frame = frame_data[frame_idx]

        logfile.write(str(frame) + "\n")

        if frame.boss_active:
            frames_without_boss = 0
        else:
            frames_without_boss += 1

        if state == State.NO_BOSS and frame.boss_active:
            if len(attempts) > 0:
                current = frame_to_ms(frame_idx)
                last = frame_to_ms(attempts[-1][1])
                # calculate how much time elapsed since the last attempt ended
                time_from_last_attempt = current - last
            else:
                # when there were no attempts yet then we don't need a cooldown
                time_from_last_attempt = MIN_MS_BETWEEN_ATTEMPTS

            if time_from_last_attempt >= MIN_MS_BETWEEN_ATTEMPTS:
                last_attempt_start = frame_idx
                state = State.ATTEMPT_STARTED

        if state == State.ATTEMPT_STARTED and frame.you_died_visible:
            end = frame_idx
            attempts.append((last_attempt_start, end, frame.boss_hp_pct))
            state = State.NO_BOSS

    logfile.close()

    print()
    for idx, attempt in enumerate(attempts):
        hp = round(attempt[2], 1)
        start = frame_to_ms(attempt[0])
        end = frame_to_ms(attempt[1])
        length = (end - start) / 1000

        print(f"Attempt #{idx + 1}: {hp}%, took {length}ms ({start}-{end}ms)")


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
