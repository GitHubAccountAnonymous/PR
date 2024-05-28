from abc import ABC, abstractmethod
from collections import defaultdict
from data.build.filters import YouTubeTextAvailFilter, YouTubeTextQualityFilter, YouTubeTextAppropFilter, \
    YouTubeAudioMusicFilter, YouTubeAudioLangFilter
import librosa
import logging
import numpy as np
import os
import random
import re
import shutil
import soundfile as sf
from tqdm import tqdm
from utils.online import get_yt
from utils.os import list_visible, read, read_lines, remove, write_lines
from utils.misc import summarize_dataset
from utils.text import standard_text, yt_subtitle_lines

logger = logging.getLogger(__name__)


class BaseDatasetBuilder(ABC):
    def __init__(self, path, config):
        """
        :param path: `str`. Path to the dataset to be built.
        :param config: `dict`. Configuration dictionary.
        """
        super(BaseDatasetBuilder, self).__init__()
        self.path = path
        self.config = config

    @abstractmethod
    def build(self):
        raise NotImplementedError


class YouTubeDatasetBuilder(BaseDatasetBuilder):

    def build(self):
        os.makedirs(self.path, exist_ok=True)
        open(os.path.join(self.path, '.build'), 'w').close()
        if os.path.isfile(os.path.join(self.path, 'videoids.txt')):
            videoids = read_lines(os.path.join(self.path, 'videoids.txt'))
        else:
            videoids = get_yt(self.config['filter']['code'])
            write_lines(videoids, os.path.join(self.path, 'videoids.txt'))

        filters = [YouTubeTextAvailFilter(self.path, self.config['filter']['text']),
                   YouTubeTextQualityFilter(self.path, self.config['filter']['text']),
                   YouTubeTextAppropFilter(self.path, self.config['filter']['text'], self.config['device']),
                   YouTubeAudioMusicFilter(self.path, self.config['filter']['audio'], self.config['device']),
                   YouTubeAudioLangFilter(self.path, self.config['filter']['audio'], self.config['device'])]
        for filt in filters:
            videoids = filt(videoids)

        self._make_utts(videoids)
        # After getting RuntimeError from the end of _make_utts(), create file indicating ambiguous utterances as prompted,
        # comment out all preceding lines in build(), and run the build_dataset program again
        self._make_splits()
        self._resample()
        summarize_dataset(self.path)

    def _filter_seqint(self, subs):
        """
        Filters out sequential integers littering some subtitles. For example, `subs` as output by `yt_subtitle_lines`
        for YouTube video with ID NsWFRvL3Y1Q would look like:
        ---
        0
        00:00:00.020 --> 00:00:04.790
        You now have a plan of action for that first date. We've explored the three dating markets.
        1
        00:00:04.790 --> 00:00:06.620
        Now you got some opportunities lined up.
        2
        00:00:06.920 --> 00:00:10.820
        to actually put these skills into action. And what skills are these Johnny?
        ---
        This function filters out the junk lines '0', '1', '2', etc.
        :param subs: `list`, as output by `yt_subtitle_lines`, containing subtitle lines. The first element should be a
        string representation of an integer, which would qualify `subs` as a legitimate target of this function.
        :return: `list` containing subtitle lines with sequential integer lines removed.
        """
        filtered = []
        last = int(subs[0]) - 1
        for line in subs:
            if line.isdigit():
                current = int(line)
                if current == last + 1:
                    last = current
                    continue
            filtered.append(line)
        return filtered

    def _make_splits(self):
        assert set(self.config['splits'].keys()) == {'train', 'dev', 'test', 'test-amb'}
        random.seed(0)
        utt2spk = read_lines(os.path.join(self.path, 'utts', 'utt2spk'))
        uttids = [line.split(' ')[0] for line in utt2spk]
        utt2spk = {line.split(' ')[0]: line.split(' ')[1] for line in utt2spk}
        spk2utt = {}
        for uttid, spkid in utt2spk.items():
            if spkid not in spk2utt:
                spk2utt[spkid] = []
            spk2utt[spkid].append(uttid)

        random.shuffle(uttids)
        uttids_ambig = set(read_lines(os.path.join(self.path, 'ambig.txt')))
        uttids = set(uttids).difference(uttids_ambig)
        total_utts = len(uttids) + len(uttids_ambig)

        splits = {}
        N_target = {}
        for split in self.config['splits']:
            splits[split] = []
            os.makedirs(os.path.join(self.path, split), exist_ok=True)
            N_target[split] = round(total_utts * self.config['splits'][split])
        N_target['train'] = total_utts - sum([N_target[split] for split in self.config['splits'] if split != 'train'])

        amb_ratio = {spk: len([uttid for uttid in spk2utt[spk] if uttid in uttids_ambig]) / len(spk2utt[spk]) for spk in spk2utt}
        amb_ratio = {k: v for k, v in sorted(amb_ratio.items(), key=lambda item: -item[1])}
        probs = {'test-amb': 0.7, 'train': 0.1, 'dev': 0.1, 'test': 0.1}

        for spk in amb_ratio:
            k = list(probs.keys())
            p = [probs[key] for key in k]
            total_p = sum(p)
            p = [prob / total_p for prob in p]
            k = np.random.choice(k, p=p)
            splits[k] += spk2utt[spk]
            if len(splits[k]) >= N_target[k] and k != 'train':
                del probs[k]

        assert sum([len(splits[k]) for k in splits]) == len({uttid for k in splits for uttid in splits[k]})
        for split in splits:
            logging.info('Organizing ' + split + ' utterance files')
            splits[split].sort()
            new_audio_path = os.path.join(self.path, split, 'audio')
            new_text_path = os.path.join(self.path, split, 'text')
            os.makedirs(new_audio_path, exist_ok=True)
            os.makedirs(new_text_path, exist_ok=True)
            for uttid in tqdm(splits[split]):
                wav = uttid + '.wav'
                txt = uttid + '.txt'
                shutil.move(os.path.join(self.path, 'utts', 'audio', wav), os.path.join(new_audio_path, wav))
                shutil.move(os.path.join(self.path, 'utts', 'text', txt), os.path.join(new_text_path, txt))

        logging.info('Writing utt2spk for each split')
        splits = set(list_visible(self.path, 'd')).difference({'utts'})
        for split in splits:
            split_path = os.path.join(self.path, split)
            audio_path = os.path.join(split_path, 'audio')
            text_path = os.path.join(split_path, 'text')
            audio_uttids = [file[:file.find('.wav')] for file in list_visible(audio_path, 'f')]
            text_uttids = [file[:file.find('.txt')] for file in list_visible(text_path, 'f')]
            audio_uttids.sort()
            text_uttids.sort()
            assert audio_uttids == text_uttids
            uttids = audio_uttids
            with open(os.path.join(split_path, 'utt2spk'), 'w') as f:
                for uttid in uttids:
                    f.write(uttid + ' ' + uttid[:uttid.rfind('-')] + '\n')

        remove(os.path.join(self.path, 'utts'))

    def _make_utts(self, videoids):
        """
        Creates utterances from YouTube videos, including text and audio for each utterance.
        :param videoids: `list` or `set` of YouTube video IDs. These video IDs should already have audio and subtitle
        files for the whole video.
        """
        utt_dir = os.path.join(self.path, 'utts')
        utt_audio_dir = os.path.join(utt_dir, 'audio')
        utt_text_dir = os.path.join(utt_dir, 'text')
        audio_dir = os.path.join(self.path, 'audio')
        subs_dir = os.path.join(self.path, 'subs')
        videoid2file = {file[:file.find('.')] : file for file in list_visible(subs_dir, 'f')}
        os.makedirs(utt_audio_dir, exist_ok=True)
        os.makedirs(utt_text_dir, exist_ok=True)
        timestamp_regex = r'^([0-9]+[:\.]*)* --> ([0-9]+[:\.]*)*'

        def append_group(group, subs):
            group['text'] = group['text'].strip()
            subs.append(group)
            return subs

        # Creating utterances
        utts = defaultdict(list)
        # Keys in utts will be video IDs
        # Values in utts will be a list of utterances corresponding to the video ID
        logging.info('Parsing utterances')
        for videoid in tqdm(videoids):
            subs_whole = yt_subtitle_lines(read(os.path.join(subs_dir, videoid2file[videoid])))
            if subs_whole[0].isdigit():
                subs_whole = self._filter_seqint(subs_whole)

            subs = []
            # Each element in subs will be a dictionary of the format:
            # {'time': (start, end), 'text': <text>}
            # Creating groups of subtitle lines that belong to the same timestamp interval
            time_group = None
            for i, line in enumerate(subs_whole):
                match = re.match(timestamp_regex, line)
                # If line is a timestamp line
                if match is not None:
                    line = line[:match.end()]
                    if time_group is not None:
                        subs = append_group(time_group, subs)
                    time = tuple(round(self._time_to_sec(item.strip()), 4) for item in line.split('-->'))
                    time_group = {'time': time, 'text': ''}
                else:
                    if time_group is None:
                        continue
                    line = standard_text(line)
                    time_group['text'] += line.strip() + ' '
                    if i == len(subs_whole) - 1:
                        subs = append_group(time_group, subs)

            # Clean up text a bit
            for group in subs:
                group['text'] = group['text'].strip()

            # Creating utterances based on connections between successive subtitle groups
            current = []
            for i, time_group in enumerate(subs):
                current.append(time_group)
                # If last character is period or question mark, complete formation of current utterance
                if (time_group['text'] != '' and
                    (time_group['text'][-1] == '.' or time_group['text'][-1] == '?') and
                    random.uniform(0, 1) < 0.8
                ):
                    utts[videoid].append(current)
                    current = []

        logging.info('Writing utterances')
        # Writing to utts/audio and utts/text directories
        for videoid, video_utts in tqdm(utts.items()):
            whole, sr = librosa.load(os.path.join(audio_dir, f'{videoid}.wav'), sr=None)
            for i, utt in enumerate(video_utts):
                uttid = videoid + '-' + str(i)
                text = ''
                for line in utt:
                    text += line['text'] + ' '
                text = text.strip()
                start = utt[0]['time'][0]
                end = utt[-1]['time'][1]
                if end - start < self.config['min_utt_dur'] or end - start > self.config['max_utt_dur']:
                    continue
                start = round(start * sr)
                end = round(end * sr)
                audio = whole[start:end]
                sf.write(os.path.join(utt_audio_dir, uttid + '.wav'), audio, sr)
                with open(os.path.join(utt_text_dir, uttid + '.txt'), 'w') as f:
                    f.write(text)

        # Writing utts/utt2spk
        uttids = [file[:file.find('.')] for file in list_visible(utt_audio_dir, 'f')]
        uttids.sort()
        with open(os.path.join(utt_dir, 'utt2spk'), 'w') as f:
            for uttid in uttids:
                spkid = uttid[:11]
                f.write(uttid + ' ' + spkid + '\n')

        # Remove everything in dataset directory except for utts
        for path in set(list_visible(self.path)).difference({'utts'}):
            remove(os.path.join(self.path, path))
        logging.info('MANUALLY: Place file with lines being utterance IDs corresponding to ambiguous text as ambig.txt in ' + self.path)
        logging.info('Then, run _make_splits() by commenting out all preceding lines in build()')
        raise RuntimeError

    def _resample(self):
        for split in set(list_visible(self.path, 'd')):
            logging.info('Resampling ' + split + ' audio')
            split_path = os.path.join(self.path, split, 'audio')
            files = [file for file in list_visible(split_path, 'f') if file.endswith('.wav')]
            files.sort()
            for file in tqdm(files):
                file_path = os.path.join(split_path, file)
                data, sr = librosa.load(file_path, sr=self.config['samplerate'])
                sf.write(file_path, data, self.config['samplerate'])

    def _time_to_sec(self, time):
        """
        Convert a time string of the format HH:MM:SS.SSS to a float representing the total time in seconds.
        :param time: `str` containing time in the format HH:MM:SS.SSS.
        :return: `float` representing the total time in seconds.
        """
        parts = time.split(':')
        return float(parts[2]) + 60*(int(parts[1]) + 60*int(parts[0]))


SUPPORTED_DATASET_BUILDERS = {'YouTube': YouTubeDatasetBuilder}
