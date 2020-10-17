# Few Shot Dialogue Generation using DiKTNet

This repository reflects the final version for my Dialogue Generation Project during my internship at [Saarthi.ai](https://saarthi.ai). The intermediate commits have been removed for privacy reasons.

The basic file structure of this repo is almost same as the original repository by @ishalyminov excluding some scripts.

### Additions

1. Added class `DSTCZslCorpus` in `utils/corpora.py` for corpus creation and vocabulary generation of the DSTC Dataset.

2. Added class `ZslDSTCDataLoader` in `utils/data_loaders.py` as the DataLoader for DSTC Dataset.

### Data Directory

Data must be present in the main directory (For DSTC8 Dataset, folder named (dstc) is already present. Download the dataset, and unzip in here i.e. folders train, dev, test from dataset must be here).

#### laed_features

The laed_features are added in the repository. These were generated using through the model trained during  The given `laed_features` folder has another folder named `features_dstc`. This folder has the representations (specifically for DSTC8) generated using LAED architecture. They have been added, because, they will be used as the input to train.

### Terminal Code for training FSDG
```
python train_fsdg.py \
    DSTCZslCorpus \
    --data_dir dstc/ \
    --laed_z_folders laed_features/features_dstc/ \
    --black_domains $domain \
    --black_ratio 0.9 \
    --action_match False \
    --target_example_cnt 0 \
    --random_seed $rnd `
```
