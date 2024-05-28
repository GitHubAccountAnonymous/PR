from feats.base import BaseFeaturizer
from kaldiio import ReadHelper
import logging
import math
import os
import pickle
import shutil
import subprocess
from utils.os import list_visible, read, read_lines, reset
from utils.text import remove_special

logger = logging.getLogger(__name__)


class KaldiFeaturizer(BaseFeaturizer):

    def __init__(self, config):
        """
        :param config: `dict`. Portion of configuration dictionary under `"data"`-->`"audio"`-->`"featurizer"`. This
        should contain a `"kaldi_eg"` key, whose corresponding value is the path to the Kaldi example to use.
        """
        super().__init__(config)
        self.kaldi_eg = self.config['kaldi_eg']

    def __call__(self, split_path):
        """
        Featurizes all utterances in a dataset's split using Kaldi and saves to `feat/[split]/kaldi` directory inside
        dataset.
        :param split_path: `str`. Path to a dataset's split, which should conform to standard data format (see README
        file).
        """
        self._create_data(split_path)
        split = os.path.basename(split_path)
        self.output_path = os.path.join(os.path.dirname(split_path), 'feat', split, 'kaldi')
        output_path_abs = os.path.abspath(self.output_path)

        root = os.getcwd()
        os.chdir(self.config['kaldi_eg'])
        subprocess.run(['./custom.sh', split, output_path_abs])
        os.chdir(root)

        self._split_scp(os.path.join(self.output_path, 'emb'))
        self.ali_dict, self.words_dict = self._align(self.output_path)

    def _align(self, output_path):
        """
        Performs phone-word alignment and saves the returned value to file.
        :param output_path: `str`. Path to `feat/[split]/kaldi` directory, where the Kaldi outputs are saved.
        :return: `tuple` `(phones, words)`. `phones` is a dictionary of aligned phones and words. `words` is a
        dictionary of transcribed words in each utterance.
        """
        logging.info('Time-phone-word alignment starting')
        self.word2int = self._read_words(os.path.join(self.kaldi_eg, 'data/lang/words.txt'))

        lines = read_lines(os.path.join(output_path, 'text'))
        lines = [line.split() for line in lines]
        words = {line[0]: line[1:] for line in lines}
        for k in words:
            words_list = []
            for word in words[k]:
                try:
                    word_int = self.word2int[word]
                except KeyError:
                    word_int = self.word2int['<unk>']
                words_list.append(word_int)
            words[k] = words_list

        # Reading in decode_phones.ctm
        ali_dir = os.path.join(output_path, 'ali')
        ctms = [item for item in list_visible(ali_dir, 'f') if item.endswith('.ctm')]
        ctms.sort()
        decode_phones_ctm = []
        for ctm in ctms:
            current = read_lines(ali_dir + '/' + ctm)
            decode_phones_ctm.extend(current)

        # Format of phones[utt_id]:
        # [start_time, duration, phone, word, word_idx]
        phones = {}
        for line in decode_phones_ctm:
            sep = line.find(' ')
            key = line[:sep].strip()
            value = line[sep + 1:].strip().split()
            if key not in phones:
                phones[key] = [[3 * float(value[1]), 3 * float(value[2]), int(value[3])]]
            else:
                phones[key].append([3 * float(value[1]), 3 * float(value[2]), int(value[3])])

        # Reading in phone sequence for each word
        align_lexicon = read_lines(os.path.join(self.kaldi_eg, 'data/lang_rescore/phones/align_lexicon.int'))

        word2phones = {}
        for line in align_lexicon:
            line_list = line.split()[1:]
            line_list = [int(l) for l in line_list]
            word = line_list[0]
            phone_seq = line_list[1:]
            if word not in word2phones:
                word2phones[word] = [phone_seq]
            else:
                word2phones[word].append(phone_seq)

        # Assigning words to each phone for each utterance
        for key in phones:
            word_seq = words[key]
            i = 0
            for word_i in range(len(word_seq)):
                word = word_seq[word_i]
                phone_seqs = word2phones[word]
                phone_seqs_len = set(len(seq) for seq in phone_seqs)
                seq_found = False

                while not seq_found:
                    for l in phone_seqs_len:
                        for seq in phone_seqs:
                            if len(seq) == l:
                                if [phones[key][j][2] for j in range(i, min(i + l, len(phones[key])))] == seq:
                                    for j in range(i, i + l):
                                        phones[key][j] += [word, word_i]
                                    seq_found = True
                                    i += l
                            if seq_found:
                                break
                        if seq_found:
                            break

                    if not seq_found:
                        try:
                            prev_word = phones[key][i - 1][3]
                        except IndexError:
                            prev_word = None
                        if prev_word:
                            prev_word_i = phones[key][i - 1][4]
                            phones[key][i] += [prev_word, prev_word_i]
                        i += 1

            for j in range(i, len(phones[key])):
                try:
                    prev_word = phones[key][j - 1][3]
                except IndexError:
                    prev_word = None
                if prev_word:
                    prev_word_i = phones[key][j - 1][4]
                    phones[key][j] += [prev_word, prev_word_i]

        with open(os.path.join(output_path, 'ali', 'time-phone-word.pkl'), 'wb') as f:
            pickle.dump((phones, words), f)
        logging.info('Time-phone-word alignment complete')
        return phones, words

    def _create_data(self, src):
        """
        Creates the directory within data/ as required by Kaldi.
        :param src: `str`. Path to source dataset, which should conform to standard data format (see README file).
        :param kaldi_eg: `str`. Path to a Kaldi example directory, e.g. kaldi/egs/tedlium/s5_r3/. This directory should
        contain a `data/` subdirectory.
        :return: None.
        """
        split = 'custom_' + os.path.basename(src)
        reset(os.path.join(self.kaldi_eg, 'data', split), 'd')
        shutil.copy(os.path.join(src, 'utt2spk'), os.path.join(self.kaldi_eg, 'data/' + split))
        os.chmod(os.path.join(self.kaldi_eg, 'data/' + split + '/utt2spk'), 0o666)
        self._create_wavscp(os.path.join(src, 'audio'), os.path.join(self.kaldi_eg, 'data/' + split))
        self._create_text(os.path.join(src, 'text'), os.path.join(self.kaldi_eg, 'data/' + split))

    def _create_wavscp(self, audio_path, save):
        """
        Creates the wav.scp file as required by Kaldi.
        :param audio_path: `str`. Path to `audio/` directory, which should conform to standard data format (see README
        file).
        :param save: `str`. Directory in which to save the created `wav.scp` file.
        :return: None
        """
        wavs = list_visible(audio_path, 'f')
        wavs.sort()
        assert all([filename.endswith('.wav') for filename in wavs])
        with open(os.path.join(save, 'wav.scp'), 'w') as f:
            for filename in wavs:
                path = os.path.abspath(os.path.join(audio_path, filename))
                f.write(filename[:-4] + ' ' + path + '\n')

    def _create_text(self, text_path, save):
        """
        Creates the text file required by Kaldi.
        :param text_path: `str`. Path to `text/` directory, which should conform to standard data format (see README
        file).
        :param save: `str`. Directory in which to save the created `text` file.
        :return: None
        """
        txts = list_visible(text_path, 'f')
        txts.sort()
        assert all([filename.endswith('.txt') for filename in txts])
        with open(os.path.join(save, 'text'), 'w') as text:
            for filename in txts:
                single_text = read(os.path.join(text_path, filename)).strip().lower()
                single_text = remove_special(single_text, [])
                text.write(filename[:-4] + ' ' + single_text + '\n')

    def _read_words(self, path):
        """
        Parses a words.txt file from Kaldi (such as that within a data/lang/ directory in a Kaldi example) into a dictionary
        mapping words to integers.
        :param path: `str`. Path to file whose lines should be of the format `<word> <int>`.
        :return: `dict` with words as keys and integers as values.
        """
        wordints = read_lines(path)
        wordints = [line.split() for line in wordints]
        word2int = {line[0]: int(line[1]) for line in wordints}
        return word2int

    def _split_scp(self, dir):
        """
        Splits lines in .scp files into multiple files, each containing a specified number of lines. Note that
        output.scp, the file containing all lines from all .scp files, is not split.
        :param dir: `str`. Path to directory containing .scp files.
        :return: None.
        """
        self.split_dir = os.path.join(dir, 'split_scp')
        reset(self.split_dir, 'd')
        scps = [file for file in list_visible(dir, 'f') if file.endswith('.scp')]
        try:
            scps.remove('output.scp')
        except ValueError:
            pass

        for scp in scps:
            shutil.copyfile(os.path.join(dir, scp), os.path.join(self.split_dir, scp))
            lines = read_lines(os.path.join(self.split_dir, scp))

            pieces = math.ceil(len(lines) / self.config['max_lines_per_scp'])
            for i in range(pieces):
                start = i * self.config['max_lines_per_scp']
                end = start + self.config['max_lines_per_scp']
                new_filename = scp.replace('.scp', '_' + str(i) + '.scp')
                with open(os.path.join(self.split_dir, new_filename), 'w') as f:
                    for j in range(start, end):
                        try:
                            line = lines[j]
                        except IndexError:
                            assert i == pieces - 1
                            break
                        old_dir = os.path.dirname(line.split(' ')[1])
                        new_dir = os.path.join(self.output_path, 'emb')
                        line = line.replace(old_dir, new_dir)
                        f.write(line + '\n')

            os.remove(os.path.join(self.split_dir, scp))

    def read_kaldi(self, filename, filetype):
        """
        Reads Kaldi data.
        :param filename: `str`. Name of file.
        :param filetype: `str`. Type of file, e.g. `"ark"` or `"scp"`.
        :return: Kaldi file contents.
        """
        contents = {}
        helper = ReadHelper(filetype + ':' + filename)
        for k, v in helper:
            contents[k] = list(v)
        return contents


SUPPORTED_AUDIO_FEATURIZERS = {
    "Kaldi": KaldiFeaturizer
}