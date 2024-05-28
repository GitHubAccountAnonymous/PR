from feats.audio import SUPPORTED_AUDIO_FEATURIZERS
from feats.base import BaseFeaturizer
from feats.text import SUPPORTED_TEXT_FEATURIZERS
import kaldiio
import logging
import numpy as np
import os
import pickle
from utils.misc import find_match
from utils.os import list_visible, read, read_lines, reset
from utils.text import get_punctuation, remove_special

logger = logging.getLogger(__name__)


class MultimodalFeaturizer(BaseFeaturizer):
    def __init__(self, config):
        """
        Initializes a featurizer for audio and text modalitites.
        :param config: `dict`. Configuration dictionary under `"data"`. This should contain both `"audio"` and`"text"`
        keys.
        """
        super().__init__(config)
        self.audio_config = self.config['audio']['featurizer']
        self.text_config = self.config['text']['featurizer']
        self.audio_featurizer = SUPPORTED_AUDIO_FEATURIZERS[self.audio_config['name']](self.audio_config)
        self.text_featurizer = SUPPORTED_TEXT_FEATURIZERS[self.text_config['name']](self.text_config)


class MultimodalFeaturizerKaldi(MultimodalFeaturizer):

    def __call__(self, split_path):
        self.split_path = split_path
        self.split = os.path.basename(self.split_path)
        self.output_path = os.path.join(os.path.dirname(self.split_path), 'feat', self.split, 'concat')
        self.audio_featurizer(self.split_path)
        self.concat_emb(self.audio_featurizer, self.text_featurizer)
        self.make_examples()

    def concat_emb(self, audio_featurizer, text_featurizer):
        self.concatenated = 0
        reset(self.output_path, 'd')
        open(os.path.join(self.output_path, 'utt2spk'), 'w').close()
        scps = [file for file in list_visible(audio_featurizer.split_dir, 'f') if file.endswith('.scp')]
        scps.sort()
        for a in scps:
            logging.info('Processing ' + a)
            ark = audio_featurizer.read_kaldi(os.path.join(audio_featurizer.split_dir, a), 'scp')
            kaldi_dict = {}
            for i, k in enumerate(ark):
                if i % 1000 == 0:
                    logging.info('Processing ' + a + ' | Stacking ' + str(i+1) + '/' + str(len(ark)))
                kaldi_dict[k] = np.vstack(ark[k])
                self.audio_dim = kaldi_dict[k].shape[1]
            del ark

            emb_dict, emb_words_dict, emb_labels_dict = self.concat_one_ark(audio_featurizer, text_featurizer, kaldi_dict)
            f = open(os.path.join(self.output_path, 'utt2spk'), 'a')
            out_ark = {}
            for i, k in enumerate(emb_dict):
                if i % 1000 == 0:
                    logging.info('Processing ' + a + ' | Utterance ' + str(i+1) + '/' + str(len(emb_dict)))
                emb = emb_dict[k]
                labels = emb_labels_dict[k]
                digits = len(str(emb.shape[0]))
                for i in range(emb.shape[0]):
                    utt_id = k + '-' + str(i).zfill(digits)
                    out_ark[utt_id] = emb[i, :]
                    f.write(utt_id + ' ' + str(labels[i]) + '\n')
            f.close()

            logging.info('Saving embeddings for ' + a)
            new_name = a.replace('output', 'emb')
            ark_name = new_name.replace('.scp', '.ark')
            kaldiio.save_ark(os.path.join(self.output_path, ark_name), out_ark,
                             scp=os.path.join(self.output_path, new_name))
            new_name = a.replace('output', 'words').replace('scp', 'dict')
            with open(os.path.join(self.output_path, new_name), 'wb') as f:
                pickle.dump(emb_words_dict, f)

        del out_ark
        scps = [file for file in list_visible(os.path.join(self.output_path), 'f') if '.scp' in file]
        combined = ''
        for scp in scps:
            combined += read(os.path.join(self.output_path, scp))
        with open(os.path.join(self.output_path, 'emb.scp'), 'w') as f:
            f.write(combined)

    def concat_one_ark(self, audio_featurizer, text_featurizer, kaldi_dict):
        embeddings_dict = {}
        embeddings_words_dict = {}
        embeddings_labels_dict = {}
        total_utts = len(audio_featurizer.words_dict)
        fail_count = 0

        for k in kaldi_dict.keys():
            if self.concatenated % 100 == 0:
                logging.info('Processing utterance ' + str(self.concatenated + 1) + '/' + str(total_utts))

            try:
                ali = audio_featurizer.ali_dict[k]
                kaldi = kaldi_dict[k]
            except KeyError:
                fail_count += 1
                continue

            text = read(os.path.join(self.split_path, 'text', k + '.txt')).strip()
            text = remove_special(text, ignore=['.', ',', '?'])
            labels = get_punctuation(text)
            text = text.lower()
            text = remove_special(text, [])
            words_list = text.split()

            if len(words_list) == 0:
                fail_count += 1
                continue

            bert_words = text_featurizer(text)
            self.text_dim = len(bert_words[0])
            # print('Kaldi-BERT words matching is starting')
            kaldi_words = audio_featurizer.words_dict[k]

            words_list_tmp = []
            for word in words_list:
                try:
                    word_int = audio_featurizer.word2int[word]
                except KeyError:
                    word_int = audio_featurizer.word2int['<unk>']
                words_list_tmp.append(word_int)
            words_list = words_list_tmp

            # Dictionary mapping indices in kaldi_words to indices in words_list
            kaldi_words_tmp = [word for word in kaldi_words]
            words_list_tmp = [word for word in words_list]

            kaldi2words = {}
            kaldi_first = 0
            words_first = 0
            i = 1
            while i <= max(len(kaldi_words_tmp), len(words_list_tmp)):
                kaldi_check = kaldi_words_tmp[:i]
                words_check = words_list_tmp[:i]
                match = find_match(kaldi_check, words_check)
                if match:
                    kaldi2words[kaldi_first + match[0]] = words_first + match[1]
                    kaldi_first += match[0] + 1
                    kaldi_words_tmp = kaldi_words_tmp[match[0] + 1:]
                    words_first += match[1] + 1
                    words_list_tmp = words_list_tmp[match[1] + 1:]
                    i = 1
                else:
                    i += 1

            missing = [i for i in range(len(kaldi_words)) if i not in kaldi2words]
            if len(missing) != 0:
                fail_count += 1
                continue
            # print('Kaldi-BERT words matching is ending')

            # print('Embedding concatenation is starting')
            # This loop determines the size of embeddings to allocate
            total_rows = 0
            for seg in ali:
                if len(seg) == 5:
                    start = round(seg[0] * 100)
                    end = round(seg[1] * 100) + start
                    end = min(end, kaldi.shape[0])
                    current_rows = end - start
                    if current_rows <= 0:
                        continue
                elif len(seg) == 3:
                    pass
                    current_rows = 0
                else:
                    raise RuntimeError('Length of seg in ali is neither 3 nor 5')
                total_rows += current_rows

            if total_rows == 0:
                continue

            embeddings = np.ndarray((total_rows, self.audio_dim + self.text_dim))
            embeddings_words = []
            embeddings_labels = []

            next_empty_row_idx = 0

            fail = False
            for i, seg in enumerate(ali):

                if len(seg) == 5:
                    start = round(seg[0] * 100)
                    end = round(seg[1] * 100) + start
                    end = min(end, kaldi.shape[0])
                    current_rows = end - start
                    if current_rows <= 0:
                        continue

                    word_idx = kaldi2words[seg[4]]
                    bert_embed = bert_words[word_idx]
                    bert_embed = np.reshape(bert_embed, (1, self.text_dim))
                    kaldi_embed = kaldi[start:end, :]
                    bert_embed = np.repeat(bert_embed, kaldi_embed.shape[0], axis=0)
                    embed = np.hstack((bert_embed, kaldi_embed))

                    embeddings[next_empty_row_idx: next_empty_row_idx + embed.shape[0]] = embed
                    next_empty_row_idx += embed.shape[0]

                    try:
                        embeddings_words += [(words_list[word_idx], seg[4], labels[word_idx]) for _ in range(embed.shape[0])]
                    except (IndexError, KeyError):
                        fail = True
                        break
                    embeddings_labels += [labels[word_idx] for _ in range(embed.shape[0])]

                elif len(seg) == 3:
                    pass

                else:
                    raise RuntimeError('Length of seg in ali is neither 3 nor 5')

            if fail:
                fail_count += 1
                continue

            assert embeddings.shape[0] <= kaldi.shape[0]
            assert embeddings.shape[0] == len(embeddings_words)

            # print('Embedding concatenation is ending')

            # Each value in embeddings_words_dict is a tuple (word, index, label)
            embeddings_dict[k] = embeddings
            embeddings_words_dict[k] = embeddings_words
            embeddings_labels_dict[k] = embeddings_labels
            self.concatenated += 1

        logging.info('Successfully aligned ' + str(len(embeddings_dict)) + ' utterances, failed ' + str(fail_count))
        return embeddings_dict, embeddings_words_dict, embeddings_labels_dict

    def make_examples(self):
        os.mkdir(os.path.join(self.output_path, 'egs'))
        os.mkdir(os.path.join(self.output_path, 'egs_txt'))
        off_center = int(self.config['audio']['context'] / 2)
        words_dict = {}
        dicts = [f for f in list_visible(self.output_path, 'f') if '.dict' in f]
        for file in dicts:
            with open(os.path.join(self.output_path, file), 'rb') as f:
                current = pickle.load(f)
            for k in current:
                words_dict[k] = current[k]

        scps = [f for f in list_visible(self.output_path, 'f') if f.endswith('.scp')]
        scps.remove('emb.scp')
        scps.sort()
        for scp in scps:
            logging.info('Reading data of ' + scp)
            embed = self.audio_featurizer.read_kaldi(os.path.join(self.output_path, scp), 'scp')
            uttids = {}
            for i, k in enumerate(embed):
                k_rev = k[::-1]
                last_dash_idx = len(k) - k_rev.find('-') - 1
                uttid = k[:last_dash_idx]
                if uttid not in uttids:
                    uttids[uttid] = []
                uttids[uttid].append(k)

            # To write .feat file for the current .scp
            for j, uttid in enumerate(uttids):
                embed_keys = uttids[uttid]
                digits = len(str(len(embed_keys)))
                for i in range(len(embed_keys)):
                    assert embed_keys[i] == uttid + '-' + str(i).zfill(digits)
                feats = np.vstack([embed[k] for k in embed_keys])
                assert len(words_dict[uttid]) == feats.shape[0]
                with open(os.path.join(self.output_path, 'egs', uttid + '.feat'), 'wb') as f:
                    pickle.dump(feats, f)

            egs_f_name = scp.replace('.scp', '.txt')
            logging.info('Writing to ' + egs_f_name)
            egs_f = open(os.path.join(self.output_path, 'egs_txt', egs_f_name), 'w')
            # Each line in this file is of the format:
            # <utterance-id> <start> <label>
            # <start> indicates the first 1792-vector to be included from the features.
            # Including <start>, 301 1792-vectors will be used in the training example.
            # <start> and <end> are zero indexed
            # <label> indicates the punctuation at the center of start to end
            for uttid in uttids:
                words = words_dict[uttid]
                prev_word_idx = 0
                for i in range(len(words)):
                    word_idx = words[i][1]
                    if word_idx != prev_word_idx:
                        assert i != 0
                        egs_f.write(uttid + ' ' + str(i - off_center - 1) + ' ' + str(words[i - 1][2]) + '\n')
                    prev_word_idx = word_idx
                egs_f.write(uttid + ' ' + str(i - off_center) + ' ' + str(words[-1][2]) + '\n')
            egs_f.close()

        egs_txt_files = list_visible(os.path.join(self.output_path, 'egs_txt'), 'f')
        egs_txt_files.sort()

        big_egs_txt_f = open(os.path.join(self.output_path, 'egs_txt', 'egs.txt'), 'w')
        for file in egs_txt_files:
            lines = read_lines(os.path.join(self.output_path, 'egs_txt', file))
            for line in lines:
                big_egs_txt_f.write(line + '\n')
        big_egs_txt_f.close()
