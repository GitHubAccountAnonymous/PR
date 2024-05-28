#!/bin/bash
KALDI_PATH=kaldi/egs/tedlium/s5_r3

unzip extras/kaldi-emb.zip -d extras/
if [ -d extras/__MACOSX ]; then
    rm -r extras/__MACOSX
fi

for item in steps utils conf; do
  if [ -d $KALDI_PATH/$item ]; then
    rm -r $KALDI_PATH/$item
  fi
done

if [ -f $KALDI_PATH/cmd.sh ]; then
  rm $KALDI_PATH/cmd.sh
fi

mv extras/kaldi-emb/* $KALDI_PATH/
rm -r extras/kaldi-emb