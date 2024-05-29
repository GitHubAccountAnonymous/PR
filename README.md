# Punctuation Restoration


## Installation

Install this framework by using
```bash
git clone https://github.com/GitHubAccountAnonymous/PunctuationRestoration
bash setup.sh
```
Please install the following extra items as needed.

### Kaldi
1. In `PunctuationRestoration/`, run
```bash
git clone https://github.com/kaldi-asr/kaldi
```
and proceed with installing [Kaldi](https://github.com/kaldi-asr/kaldi).
2. Download an additional zip file from [this Google Drive link](https://drive.google.com/file/d/1yfxuqtXrFMi1GhDl9dDxhHbVQE6-tXlf/view?usp=sharing) and place it inside `extras/`. Then, run:
```bash
bash extras/kaldi_setup.sh
```

### Pretrained Models
Depending on what models you need, pretrained ones are available for download [here](https://drive.google.com/drive/folders/1YospBmQgXOWE3C5PexAm_3UeJnU1HMXD?usp=sharing). Please place them in the same directory under `models/` as found in the download folder.


## Data

This section is about the `data/` directory formatting and its contents.

Datasets should be placed inside `data/` as subdirectories. Each dataset should contain its subsets, such as `train/`, `dev/`, and `test/`. However, data subsets may not be limited by these three standard ones. Please see the "Modes" subsection below for guidance on how each subset is used. Make sure that you do not have a subset named `feat/`, as this is a special keyword used in the framework to store embeddings. Each subset should then contain an `audio/` folder, a `text/` folder, and an `utt2spk` file. The `audio/` and `text/` folders should have `.wav` speech audio and `.txt` transcript files inside, respectively, and these files' names should be utterance IDs `[utt-id]`. Utterance IDs should have speaker ID `[spk-id]` as prefixes. `utt2spk` should be a text file in which each line is of the format:
```
[spk-id] [utt-id]
```
However, if you do not have speaker information, then equate `[spk-id]` to `[utt-id]`, in which case `utt2spk`'s lines should be of the format:
```
[utt-id] [utt-id]
```
`data`'s directory structure should look like this:
```
data/
|---dataset1/
|   |---train/
|   |   |---audio/
|   |   |   |---[utt-id1].wav
|   |   |   |---[utt-id2].wav
|   |   |   |---...
|   |   |---text/
|   |   |   |---[utt-id1].txt
|   |   |   |---[utt-id2].txt
|   |   |   |---...
|   |   |---utt2spk
|   |---...
|---...
```
In the code's comments, the term *standard data format* is used to refer to this directory structure.


## Main Program

To run the main program, execute `.sh` files in the `scripts/` directory. For example, to use `scripts/efficientpunct.sh`, run:
```bash
bash scripts/efficientpunct.sh &
```

The main program's behavior can be customized by modifying arguments in the `.sh` files, as well as the configuration file specified by the `config_path` argument. Arguments and configuration parameters will be checked by the main program for validity. However, even passing all validity checks does not guarantee that the program will run successfully. On the other hand, failing any validity check guarantees that the program will fail.

### Modes

This subsection describes valid values for the `mode` argument when running model-related programs.

The `mode` argument can take on the following values:
- `train`: Training the model. In this case, the `train`, `dev`, and `test` data subsets are required to be present.
- `predict`: Predicting on the test set. The `test` subset is required.
- `inspect`: Manually examining punctuation behavior on select utterances. An `inspect` subset is required. This mode is designed for handling a small number of samples.

Certain arguments (e.g. `optimizer` and `epochs`) are applicable only when `mode='train'`.

### Dataset Building

You can run
```bash
bash scripts/build_dataset.sh &
```
to build a dataset. Currently, only the YouTube protocol is supported, activated by setting `config_path` to `configs/YouTube.json`.


## Datasets

Download SponSpeech at [https://storage.googleapis.com/sponspeech/sponspeech.tar.gz](https://storage.googleapis.com/sponspeech/sponspeech.tar.gz).


## Other Notes

We recommend you to read the following notes only as you encounter issues requiring resolution or clarification. Knowing the following is not necessary for most usage of this framework to be successful.

### EfficientPunctBERT vs. BERTMLP

The difference between EfficientPunctBERT and BERTMLP is extremely subtle. In effect, they are exactly the same. However, BERTMLP can be thought of as "designed to be trained first". After BERTMLP is trained by fine-tuning BERT with a multi-layer perceptron (MLP) stacked on top for classification, its saved model can be used to extract final hidden layer embeddings. Before the initial writing of EfficientPunctBERT's code, these BERT embeddings were concatenated to Kaldi embeddings and saved to file. EfficientPunctBERT then re-extracted the BERT-only portions from the concatenated embeddings and applied the aforementioned MLP.

This describes the relationship between EfficientPunctBERT and BERTMLP. Essentially, if you want to fine-tune BERT for punctuation classification, use BERTMLP. If you already have text embeddings ready to be analyzed, use EfficientPunctBERT.
