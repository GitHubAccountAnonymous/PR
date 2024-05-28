from data.build.filters.base import YouTubeContentFilter
import librosa
import logging
import os
import subprocess
from tqdm import tqdm
from transformers import pipeline
from utils.os import list_visible

logger = logging.getLogger(__name__)


class YouTubeAudioContentFilter(YouTubeContentFilter):

    def __init__(self, path, config, device):
        super().__init__(path, config)
        self.content_dir = os.path.join(self.path, 'audio')
        self.sr = 16000  # sample rate
        self.pipeline_device = self._get_pipeline_device(device)

    def _download_content(self, videoids):
        os.makedirs(self.content_dir, exist_ok=True)
        done = [item[:item.find('.')] for item in list_visible(self.content_dir, 'f')]
        todo = set(videoids).difference(set(done))
        if len(todo) != 0:
            logging.info('Downloading audio')
        for videoid in tqdm(todo):
            url = f'https://www.youtube.com/watch?v={videoid}'
            command = f'yt-dlp -o "%(id)s.%(ext)s" -P {self.content_dir} -x --audio-format wav {url}'
            subprocess.run(command, shell=True, text=True)

    def _get_pipeline_device(self, device):
        if device == 'cpu':
            return -1
        elif device == 'cuda':
            return 0
        elif 'cuda:' in device:
            device_num = int(device.split(':')[1])
            assert device_num >= 0
            return device_num
        else:
            raise ValueError('Unsupported value for argument `device`')

    def _read_content(self, path):
        audio, _ = librosa.load(path, sr=self.sr)
        return audio


class YouTubeAudioMusicFilter(YouTubeAudioContentFilter):

    def __init__(self, path, config, device):
        super().__init__(path, config, device)
        self.msg = 'Filtering out YouTube videos containing primarily music, rather than speech'
        self.abbrev = 'YTAMF'
        self.pipe = pipeline("audio-classification",
                             model="FerhatDk/wav2vec2-base_music_speech_both_classification",
                             device=self.pipeline_device)

    def _criteria(self, content):
        """
        Determines whether the YouTube audio has satisfactory quality. Note that this determination is made using a
        model and could be wrong.
        :param content: `np.ndarray` of shape `(n_samples, )` where `n_samples` is the number of samples in the audio.
        :return: `bool`. `True` if the audio is satisfactory, `False` otherwise.
        """
        music_score = 0
        speech_score = 0
        start = 0
        n_samples = content.shape[0]  # total number of samples in audio
        while start < n_samples:
            end = min(start + self.sr * self.config['music_detection_dur'], n_samples)
            # if segment is less than 1 s
            if end - start < self.sr:
                break
            pred = self.pipe(content[start:end])
            for item in pred:
                score = item['score'] * (end - start)
                if item['label'] == 'music':
                    music_score += score
                elif item['label'] == 'speech':
                    speech_score += score
                elif item['label'] == 'speech_music':
                    # speech overlayed with music still has validity as an utterance in ASR dataset
                    music_score += 0.3 * score
                    speech_score += 0.7 * score
            start = end

        if music_score > speech_score:
            return False
        else:
            return True


class YouTubeAudioLangFilter(YouTubeAudioContentFilter):

    def __init__(self, path, config, device):
        super().__init__(path, config, device)
        self.msg = 'Filtering out YouTube videos whose primary language is not English'
        self.abbrev = 'YTALF'
        self.pipe = pipeline('audio-classification',
                             model='sanchit-gandhi/whisper-small-ft-common-language-id',
                             device=self.pipeline_device)

    def _criteria(self, content):
        """
        Determines whether the YouTube audio's primary language is English. Note that this determination is made using a
        model and could be wrong.
        :param content: `np.ndarray` of shape `(n_samples, )` where `n_samples` is the number of samples in the audio.
        :return: `bool`. `True` if the audio's primary language is English, `False` otherwise.
        """
        pred = self.pipe(content)
        if pred[0]['label'] == 'English':
            return True
        else:
            return False
