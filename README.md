# Self-supervised Text Style Transfer using StarGAN
This repository contains the code for the paper [XXX](https://dl.acm.org/doi/10.1145/3678179), produced for the PoliTo course of Deep Natural Language Processes.

It includes the Python package to train and test the StarGAN architecture for Text Style Transfer described in the paper.

## Installation
The following command will clone the project:
```
git clone -b StarGAN --single-branch https://github.com/QuantDreamGit/NLP-project
```

To install the required libraries and dependencies, you can refer to the `env.yml` file.

## Usage
The package provides the scripts to implement, train and test the StarGAN architecture for Text Style Transfer described in the paper.

Specifically, we focus on *sentiment* (positive, neutral and aggressive) transfer tasks.

## Data

### Sentiment transfer
We use the [Yelp](https://papers.nips.cc/paper_files/paper/2017/hash/2d2c8394e31101a261abf1784302bf75-Abstract.html) dataset following the same splits as in [Li et al.](https://aclanthology.org/N18-1169/) available in the official [repository](https://github.com/lijuncen/Sentiment-and-Style-Transfer). Put it into the `data/yelp` folder and please name the files as `[train|dev|test].[0|1].txt`, where 0 is for negative sentiment and 1 is for positive sentiment.

## Training
You can train the proposed StarGAN architecture for Text Style Transfer using the `train.py` script. It can be customized using several command line arguments such as:
- n_styles: set the total number of styles
- generator_model_tag: tag or path of the generator model
- discriminator_model_tag: tag or path of the discriminator model
- lambdas: loss weighting factors in the form "λ1|λ2|λ3|λ4|λ5" for generator (adversarial), generator (style), cycle-consistency, discriminator (real/fake) and discriminator (style), respectively
- path_db_tr: path to the training dataset 
- path_db_eval: path to the validation dataset
- path_to_references: a list of path to the references dataset (one for each style)
- learning_rate, epochs, batch_size: learning rate, number of epochs and batch size for model training

As an example, to train the CycleGAN architecture for formality transfer using our custom dataset (*Sentiment* domain), you can use the following command:
```
!CUDA_VISIBLE_DEVICES=0 python NLP-project/train.py --n_styles=3 --lang=en \
                       --path_db_tr=./data/yelp_and_aggression/yelp_and_aggression_60kTAG.txt --path_db_eval=./data/yelp_and_aggression/validation600.txt \
                       --path_to_references ./data/yelp_and_aggression/tutte_AGGRESIVE.txt ./data/yelp_and_aggression/tutte_NEGATIVE.txt ./data/yelp_and_aggression/tutte_POSITIVE.txt \
                       --shuffle \
                       --generator_model_tag=google-t5/t5-small --discriminator_model_tag=distilbert-base-cased \
                       --lambdas="1|1|10|1|1" --epochs=10 --learning_rate=5e-5 --max_sequence_length=64 --batch_size=64  \
                       --save_base_folder=./ckpts/ --save_steps=1 --eval_strategy=epochs --eval_steps=1  --pin_memory --use_cuda_if_available 
```

## Testing
Once trained, you can evaluate the performance on the test set of the trained models using the `test.py` script. It can be customized using several command line arguments such as:
- n_styles: set the total number of styles
- generator_model_tag: tag or path of the generator model
- discriminator_model_tag: tag or path of the discriminator model
- from_pretrained: folder to use as base path to load the model checkpoint(s) to test
- path_db: path to the test dataset
- path_to_references: a list of path to the references dataset (one for each style)

As an example, to test the trained models for formality transfer using the GYAFC dataset (*Family & Relationships* domain), you can use the following command:
```
CUDA_VISIBLE_DEVICES=0 python NLP-project/train.py --n_styles=3 --lang=en \
                       --path_db_test=./data/yelp_and_aggression/validation600.txt \
                       --path_to_references ./data/yelp_and_aggression/tutte_AGGRESIVE.txt ./data/yelp_and_aggression/tutte_NEGATIVE.txt ./data/yelp_and_aggression/tutte_POSITIVE.txt \
                       --generator_model_tag=google-t5/t5-large --discriminator_model_tag=distilbert-base-cased \
                       --from_pretrained=./ckpts/ --max_sequence_length=64 --batch_size=16 --pin_memory --use_cuda_if_available 
```

## License

<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-sa/4.0/88x31.png" /></a><br />This work is licensed under the <a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/">Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License</a>.

## Authors
Matteo Tomatis, Giuseppe Priolo, Federico D'Agostino

### Corresponding author
For any questions about the StarGAN Model or the implementation, you can contact us at: `s[DOT]334271[AT]studenti[DOT]polito[DOT]it`.

## Citation
If you find this work useful, please cite our paper.
