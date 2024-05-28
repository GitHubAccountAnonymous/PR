from collections import defaultdict
from data.build.filters.base import BaseDatasetFilter, YouTubeContentFilter
from detoxify import Detoxify
import logging
import os
import re
import shutil
import subprocess
from tqdm import tqdm
from utils.os import list_visible, read, write_lines
from utils.text import yt_subtitle_lines

logger = logging.getLogger(__name__)


class YouTubeTextAvailFilter(BaseDatasetFilter):

    def __init__(self, path, config):
        super().__init__(path, config)
        self.videoids = None
        self.abbrev = 'YTTAvF'

    def __call__(self, videoids):
        """
        Filters out YouTube videos without English subtitles.
        :param videoids: `list` of YouTube video IDs.
        :return: `list` containing subset of video IDs in `samples`, which have English subtitles.
        """
        logging.info('Filtering out YouTube videos without English subtitles')
        videoids = set(videoids)
        has_en = []

        for videoid in tqdm(videoids):
            url = f'https://www.youtube.com/watch?v={videoid}'
            command = f'yt-dlp --list-subs --skip-download {url}'
            try:
                res = subprocess.check_output(command, shell=True, text=True).split('\n')
            except subprocess.CalledProcessError:
                continue

            try:
                subtitles = res.index('[info] Available subtitles for ' + videoid + ':')
            except ValueError:
                continue

            assert res[subtitles + 1].startswith('Language')
            for i in range(subtitles + 2, len(res)):
                line = res[i]
                if line == '':
                    break
                lang = line.split(' ')[0]
                if lang == 'en' or lang.startswith('en-'):
                    has_en.append(videoid)
                    break

        self.videoids = has_en
        self._update_files()
        self._print_stats(len(has_en), len(videoids))
        return has_en

    def _update_files(self, keep_old=True):
        """
        Updates videoids.txt at the dataset path.
        :param keep_old: `bool` of whether the old videoids.txt file should be kept in a directory called `old/`. If
        `True`, it will first be renamed to `videoids-preYTSAF.txt`, where YTSAF stands for "YouTubeSubAvailFilter."
        :return: None.
        """
        file = os.path.join(self.path, 'videoids.txt')
        if keep_old:
            directory = os.path.join(self.path, 'old')
            os.makedirs(directory, exist_ok=True)
            kept_file = os.path.join(self.path, f'videoids-pre{self.abbrev}.txt')
            os.rename(file, kept_file)
            shutil.move(kept_file, directory)
        write_lines(self.videoids, file)


class YouTubeTextContentFilter(YouTubeContentFilter):

    def __init__(self, path, config):
        super().__init__(path, config)
        self.content_dir = os.path.join(self.path, 'subs')
        self.timestamp_regex = r'^([0-9]+[:\.]*)+ --> ([0-9]+[:\.]*)+'

    def _download_content(self, videoids):
        """
        Obtains English subtitles for each YouTube video. This function does nothing if subtitles for all video IDs
        are already downloaded.
        :param videoids: `list` or `set` containing YouTube video IDs whose corresponding subtitles are desired.
        :return: None.
        """
        os.makedirs(self.content_dir, exist_ok=True)
        done = [item[:item.find('.')] for item in list_visible(self.content_dir, 'f')]
        todo = set(videoids).difference(set(done))
        if len(todo) != 0:
            logging.info('Downloading subtitles')
        for videoid in tqdm(todo):
            url = f'https://www.youtube.com/watch?v={videoid}'
            command = f'yt-dlp --write-subs --sub-langs "en.*" --skip-download -o "%(id)s.%(ext)s" -P {self.content_dir} {url}'
            subprocess.run(command, shell=True, text=True)

    def _read_content(self, path):
        return read(path)


class YouTubeTextAppropFilter(YouTubeTextContentFilter):

    def __init__(self, path, config, device):
        """
        Initializes a filter that eliminates YouTube videos with inappropriate subtitles.
        :param path: `str` containing path to the dataset to be built.
        :param config: `dict`.
        :param device: `str`. The only form of GPU supported is `'cuda'`, with specific CUDA device specifications like
        `'cuda:0'` being automatically converted into `'cuda'`. If the string does not contain `'cuda'`, CPU will be
        used.
        """
        super().__init__(path, config)
        self.msg = 'Filtering out YouTube videos with inappropriate subtitles'
        self.abbrev = 'YTTApF'
        if 'cuda' in device:
            self.device = 'cuda'
        else:
            self.device = 'cpu'
        self.model = Detoxify('unbiased', device=self.device)

    def _criteria(self, content):
        """
        Determines whether the YouTube subtitle text is appropriate (e.g. not toxic, sexually explicit, etc.)
        :param content: `str` containing subtitles for an entire YouTube video.
        :return: `bool`. `True` if the text is appropriate, `False` otherwise.
        """
        subs = yt_subtitle_lines(content)
        subs = [line for line in subs if re.match(self.timestamp_regex, line) is None]
        strikes = 0
        for i in range(len(subs) - 2):
            lines = subs[i] + ' ' + subs[i+1] + ' ' + subs[i+2]
            pred = self.model.predict(lines)
            attn_keys = ['toxicity', 'severe_toxicity', 'identity_attack', 'threat', 'sexual_explicit']
            if any(pred[k] > self.config['inapprop_threshold'] for k in attn_keys):
                strikes += 1
            if strikes >= 6:
                return False
        return True


class YouTubeTextQualityFilter(YouTubeTextContentFilter):

    def __init__(self, path, config):
        super().__init__(path, config)
        self.msg = 'Filtering out YouTube videos with poor quality subtitles'
        self.abbrev = 'YTTQF'

    def _criteria(self, content):
        """
        Determines whether the YouTube subtitle text has satisfactory quality.
        :param content: `str` containing subtitles for an entire YouTube video.
        :return: `bool`. `True` if the text is satisfactory, `False` otherwise.
        """
        to_check = [self._satisfies_min_lines,
                    self._satisfies_regular_timestamps,
                    self._satisfies_min_punct,
                    self._satisfies_non_repeat]
        content = yt_subtitle_lines(content)
        for test in to_check:
            valid = test(content)
            if not valid:
                return False
        return True

    def _satisfies_min_lines(self, subs):
        """
        Checks whether the YouTube subtitle text contains at least `self.config['min_lines']` non-empty lines.
        :param subs: `list`, as output by yt_subtitle_lines, containing subtitles for an entire YouTube video.
        :return: `bool`. `True` if the text contains at least `self.config['min_lines']` lines, `False` otherwise.
        """
        if len(subs) >= self.config['min_lines']:
            return True
        else:
            return False

    def _satisfies_min_punct(self, subs):
        """
        Checks whether the YouTube subtitle text contains sufficient punctuation marks.
        :param subs: `list`, as output by yt_subtitle_lines, containing subtitles for an entire YouTube video.
        :return: `bool`. `True` if the text contains sufficient punctuation, `False` otherwise.
        """
        subs = [line for line in subs if re.match(self.timestamp_regex, line) is None]
        punct = [',', '.', '?', '!']
        n_punct = sum([line.count(mark) for line in subs for mark in punct])
        if len(subs) == 0:
            return False
        if n_punct / len(subs) < self.config['min_punct_to_lines']:
            return False
        else:
            return True

    def _satisfies_regular_timestamps(self, subs):
        """
        Checks whether the YouTube subtitle timestamps have regular formatting, and not extraneous terms like 'align',
        'line', 'position', etc. This is done via a proxy: Regular timestamp lines should have length < 30. If a line
        has length > 30, the video is deemed unsatisfactory. Also, some subtitles have junk timestamp lines like
        `00:08:47.127--&gt;00:08:52.866` below regular timestamp lines. These are also deemed unsatisfactory.
        :param subs: `list`, as output by yt_subtitle_lines, containing subtitles for an entire YouTube video.
        :return: `bool`. `True` if the text has regular timestamps, `False` otherwise.
        """
        junk_timestamp = r'\&[a-z]+;([0-9]+[:\.]*)+'
        for line in subs:
            if re.search(self.timestamp_regex, line) is not None and len(line) > 30:
                return False
            if re.search(junk_timestamp, line) is not None:
                return False
        return True

    def _satisfies_non_repeat(self, subs):
        """
        Checks whether the YouTube subtitle text is non-repetitive, e.g. whether it incorrectly contains same phrases
        multiple times over a short time. Specifically, the text is deemed repetitive if three distinct lines are found
        to occur at least three times within a line span of `self.config['repeat_span']`.
        :param subs: `list`, as output by yt_subtitle_lines, containing subtitles for an entire YouTube video.
        :return: `bool`. `True` if the text is non-repetitive (text is not problematic), `False` otherwise.
        """
        occur = defaultdict(list)
        for i, line in enumerate(subs):
            occur[line].append(i)

        strikes = 0
        for text, nums in occur.items():
            if len(nums) <= 2:
                continue
            diffs = [nums[i+1] - nums[i] for i in range(len(nums) - 1)]
            for i in range(len(diffs) - 1):
                triple_occur_span = diffs[i] + diffs[i+1]
                if triple_occur_span < self.config['repeat_span']:
                    strikes += 1
                    break
            if strikes >= 3:
                return False
        return True
