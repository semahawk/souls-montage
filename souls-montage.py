#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import math
import argparse
from enum import Enum
import statistics

import numpy as np
import cv2
from tqdm import tqdm

from boss_config import *
from game_config import *

RESIZE_FACTOR = 0.6
MIN_MS_BETWEEN_ATTEMPTS = 5 * 1000  # 5 seconds


def hex_to_cv2_color(hex_: int):
    r = (hex_ & 0xFF0000) >> 16
    g = (hex_ & 0x00FF00) >> 8
    b = (hex_ & 0x0000FF) >> 0
    return np.array([b, g, r])


def ms_to_hms(ms: int, include_ms=False):
    s = math.floor(ms / 1000)
    m = math.floor(s / 60)
    h = math.floor(m / 60)

    if include_ms:
        return "{:02d}:{:02d}:{:02d}.{:d}".format(h, m % 60, s % 60, ms % 1000)
    else:
        return "{:02d}:{:02d}:{:02d}".format(h, m % 60, s % 60)


class FrameData:
    video_file = ""

    def __init__(self,
                 frame_idx,
                 time,
                 boss_active=False,
                 boss_hp_pct=-1,
                 you_died_visible=False,
                 win_message_visible=False):
        self.frame_idx = frame_idx
        self.time = time
        self.boss_active = boss_active
        self.boss_hp_pct = boss_hp_pct
        self.you_died_visible = you_died_visible
        self.win_message_visible = win_message_visible

    def __repr__(self):
        return "--{}-{}--{}-{}--{}--{}--".format(
            self.time,
            str(self.frame_idx).ljust(8, "-"),
            "BOSS" if self.boss_active else "----",
            str(int(self.boss_hp_pct)).rjust(3, "-")
            if self.boss_hp_pct != -1 else "---",
            "DIED" if self.you_died_visible else "----",
            "PREY" if self.win_message_visible else "----",
        )


class Attempt:

    def __init__(self, id: int, video_file: str, start_ms: int, end_ms: int,
                 victory: bool, boss_hp: int):
        assert end_ms > start_ms, "end_ms must be after start_ms"

        self.id = id
        self.video_file = video_file
        self.start_ms = start_ms
        self.end_ms = end_ms
        self.victory = victory
        self.boss_hp = boss_hp

    def __repr__(self):
        id = self.id
        hp = round(self.boss_hp, 1)
        status = "VICTORY" if self.victory else f"YOU DIED, {hp}%"
        time = round((self.end_ms - self.start_ms) / 1000, 1)
        file = os.path.basename(self.video_file)
        start = ms_to_hms(self.start_ms, include_ms=True)
        end = ms_to_hms(self.end_ms, include_ms=True)

        return f"Attempt #{id}: {status}, took {time}s ({file} {start}-{end})"


class VideoProcessor:
    _frame_data: dict[int, FrameData] = {}

    def __init__(self, boss):
        self.last_frame_idx = 0

        self.boss_config = BOSS_CONFIG[boss]
        self.game_config = GAME_CONFIG[self.boss_config["game"]]

        self.boss_name_tmpl = self.get_tmpl(self.boss_config["boss_name_tmpl"])
        self.you_died_tmpl = self.get_tmpl(self.game_config["you_died_tmpl"])
        self.win_message_tmpl = self.get_tmpl(
            self.game_config[self.boss_config["win_message"] + "_tmpl"])

    def frame_to_ms(self, frame_idx: int):
        return int(frame_idx * (1 / self.frames_per_sec) * 1000)

    def ms_to_frames(self, ms: int):
        return int(ms * (self.frames_per_sec / 1000))

    def get_tmpl(self, fname, resized=True, cv2_flag=cv2.IMREAD_UNCHANGED):
        templates_dir = self.game_config["templates_dir"]
        fname = os.path.join(templates_dir, fname)

        if not os.path.exists(fname):
            raise Exception(f"File {fname} doesn't exist!")

        tmpl = cv2.imread(fname, cv2_flag)

        if resized:
            tmpl = cv2.resize(tmpl, (0, 0), fx=RESIZE_FACTOR, fy=RESIZE_FACTOR)

        return tmpl

    def get_box(self, frame, which: str):
        height, width = frame.shape[0], frame.shape[1]

        y1 = int(height * self.game_config[f"{which}_y_start_pct"])
        y2 = int(height * self.game_config[f"{which}_y_end_pct"])
        x1 = int(width * self.game_config[f"{which}_x_start_pct"])
        x2 = int(width * self.game_config[f"{which}_x_end_pct"])

        return frame[y1:y2, x1:x2]

    def process_video(self, filename):
        cap = cv2.VideoCapture(filename)

        self.frames_per_sec = cap.get(cv2.CAP_PROP_FPS)
        self.frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        with tqdm(desc=filename, total=self.frame_count) as progress:
            frames_processed = 0

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frames = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                frame_idx = frames + self.last_frame_idx
                self._frame_data[frame_idx] = self.process_frame(
                    frame, frame_idx)
                self._frame_data[frame_idx].video_file = filename
                frames_processed += 1

                progress.update()

            self.last_frame_idx = frames_processed

        cap.release()

    def process_frame(self, orig_frame, frame_idx) -> FrameData:
        frame_data = FrameData(frame_idx,
                               ms_to_hms(self.frame_to_ms(frame_idx)))

        frame = cv2.cvtColor(orig_frame, cv2.COLOR_BGR2GRAY)
        frame = cv2.resize(frame, (0, 0), fx=RESIZE_FACTOR, fy=RESIZE_FACTOR)

        _boss_name_tmpl = cv2.cvtColor(self.boss_name_tmpl, cv2.COLOR_BGR2GRAY)
        _you_died_tmpl = cv2.cvtColor(self.you_died_tmpl, cv2.COLOR_BGR2GRAY)
        _win_message_tmpl = cv2.cvtColor(self.win_message_tmpl,
                                         cv2.COLOR_BGR2GRAY)

        you_died_box = self.get_box(frame, "you_died")
        match = cv2.matchTemplate(you_died_box, _you_died_tmpl,
                                  cv2.TM_CCOEFF_NORMED)
        if np.amax(match) > 0.5:
            frame_data.you_died_visible = True

        win_message_box = self.get_box(frame, self.boss_config["win_message"])
        match = cv2.matchTemplate(win_message_box, _win_message_tmpl,
                                  cv2.TM_CCOEFF_NORMED)
        if np.amax(match) > 0.5:
            frame_data.win_message_visible = True

        # try to find the boss' name
        boss_bar_box = self.get_box(frame, "boss_bar")
        match = cv2.matchTemplate(boss_bar_box, _boss_name_tmpl,
                                  cv2.TM_CCORR_NORMED)
        boss_active_confidence = np.amax(match)

        # and try to see if the boss' hp bar is active
        # by checking the dominant color of this area and checking if in range
        boss_hp_bar_box = self.get_box(orig_frame, "boss_hp_bar")
        boss_hp_bar_data = np.reshape(boss_hp_bar_box, (-1, 3))
        boss_hp_bar_data = np.float32(boss_hp_bar_data)

        crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        flags = cv2.KMEANS_RANDOM_CENTERS
        _, _, centers = cv2.kmeans(boss_hp_bar_data, 1, None, crit, 10, flags)

        img = np.ones((1, 1, 3), dtype=np.uint8)
        img[:, :] = centers[0]

        lowerb = hex_to_cv2_color(
            self.game_config["boss_bar_dominant_color_lower_bound"])
        upperb = hex_to_cv2_color(
            self.game_config["boss_bar_dominant_color_upper_bound"])
        mask = cv2.inRange(img, lowerb, upperb)

        # bump the confidence by a fair amount if the dominant color of the boss' hp
        # box is what we would expect when the boss is active
        if mask[0][0] == 255:
            boss_active_confidence += 0.5

        if boss_active_confidence > 0.85:
            frame_data.boss_active = True

        # calculate how much hp the boss has (if decently confident that boss is active)
        if boss_active_confidence > 0.85:
            # when YOU DIED is visible then the screen has a semi-transparent overlay
            # making the white more gray
            if frame_data.you_died_visible:
                white_lowerb, white_upperb = (50, 50, 50), (150, 150, 150)
            else:
                white_lowerb, white_upperb = (180, 180, 180), (255, 255, 255)

            # extract the white/gray tip(s)
            whites = cv2.inRange(boss_hp_bar_box, white_lowerb, white_upperb)
            h, w = boss_hp_bar_box.shape[0], boss_hp_bar_box.shape[1]

            # go from the right towards the left side of the bar, and check if we
            # encounter any white pixels
            for x in reversed(range(1, w)):
                if np.count_nonzero(whites[:, x - 1:x] > 0):
                    frame_data.boss_hp_pct = round(x / w * 100)
                    break

        return frame_data

    def process_frame_data(self):

        class State(Enum):
            NO_BOSS = 0
            ATTEMPT_STARTED = 1
            CURRENTLY_DYING = 2
            VICTORY = 3

        state: State = State.NO_BOSS
        last_attempt_start = -1
        you_died_start_frame_idx = -1
        you_died_boss_hps: list[int] = []
        attempts: list[Attempt] = []
        next_attempt_id = 1

        def add_attempt(frame: FrameData, start_frame_idx: int,
                        end_frame_idx: int, victory: bool, boss_hp: int):
            nonlocal next_attempt_id

            attempts.append(
                Attempt(next_attempt_id, frame.video_file,
                        self.frame_to_ms(start_frame_idx),
                        self.frame_to_ms(end_frame_idx), victory, boss_hp))

            next_attempt_id += 1

        logfile = open(".frame_data.log", "w")

        sorted_frames = sorted(self._frame_data.keys())
        for frame_idx in tqdm(sorted_frames, desc="Processing frame data"):
            frame = self._frame_data[frame_idx]

            logfile.write(str(frame) + "\n")

            if state == State.NO_BOSS and frame.boss_active:
                if len(attempts) > 0:
                    current = self.frame_to_ms(frame_idx)
                    last = attempts[-1].end_ms
                    # calculate how much time elapsed since the last attempt ended
                    time_from_last_attempt = current - last
                else:
                    # when there were no attempts yet then we don't need a cooldown
                    time_from_last_attempt = MIN_MS_BETWEEN_ATTEMPTS

                if time_from_last_attempt >= MIN_MS_BETWEEN_ATTEMPTS:
                    last_attempt_start = frame_idx
                    state = State.ATTEMPT_STARTED

            elif state == State.ATTEMPT_STARTED and frame.you_died_visible:
                you_died_start_frame_idx = frame_idx
                state = State.CURRENTLY_DYING

            elif state == State.ATTEMPT_STARTED and frame.win_message_visible:
                add_attempt(frame, last_attempt_start, frame_idx, True, 0)
                state = State.NO_BOSS

            elif state == State.CURRENTLY_DYING:
                if frame.you_died_visible:
                    you_died_boss_hps.append(frame.boss_hp_pct)
                else:
                    # decide the final boss hp based on what number comes up
                    # most often in the frame data while "YOU DIED" is visible
                    # this is to prevent misreads on single frames to mess with
                    # the final reading
                    final_boss_hp = statistics.mode(you_died_boss_hps)
                    you_died_boss_hps = []

                    add_attempt(frame, last_attempt_start,
                                you_died_start_frame_idx, False, final_boss_hp)
                    state = State.NO_BOSS

        logfile.close()

        return attempts


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Souls montage")
    parser.add_argument("input_videos",
                        metavar="video",
                        type=str,
                        nargs="+",
                        help="video(s) to process")
    parser.add_argument("-b",
                        "--boss",
                        type=str,
                        choices=BOSS_CONFIG.keys(),
                        required=True)

    args = parser.parse_args()

    processor = VideoProcessor(args.boss)

    for file in args.input_videos:
        processor.process_video(file)

    attempts = processor.process_frame_data()
    for attempt in attempts:
        print(attempt)
