# KMLFC
This is the repository for paper "**Learning Kernel-Modulated Neural Representation for Efficient Light Field Compression**" (**arXiv-2307.06143**).

By [Jinglei Shi](https://jingleishi.github.io/),  [Yihong Xu](https://github.com/yihongXU)  and  [Christine Guillemot](https://people.rennes.inria.fr/Christine.Guillemot/)

<[Paper link](https://arxiv.org/abs/2307.06143)>

## Dependencies
```
python==
pytorch==
torchvision==
```
## Examples
### Preparation of the dataset
Before training the network, the preparation of the dataset is as follows:
- Create a folder containing all light fields used for training and validation, and each light field is an individual folder, with all sub-apterture images named 'lf_row_column.png', where 'row' and 'column' are row and column indices.
- Create .txt files and add the names of the light field into them. A 'tri_trainslit.txt' for training set and a 'tri_testlist.txt' for test set.

### Training
Coming soon

### Evaluation


### Pretrained Models
We provide three models respectively pretrained on HCI LF dataset ([synth.pkl](https://pan.baidu.com/s/1ZAIttST3AliL87-0y3RMmQ?pwd=0003)), UCSD LF dataset ([realworld.pkl](https://pan.baidu.com/s/1Y2rfeUa6F-PW7UgTuhWoew?pwd=0004)) and EPFL LF dataset ([epfl.pkl](https://pan.baidu.com/s/1SkwXVK3uoIUvC9wj0Q2onQ?pwd=0002)), users can just download them and put them into the folder 'saved_model'.

## Citation
Please consider citing our work if you find it useful.
```
@inproceedings{shi2020learning,
    title={Learning Fused Pixel and Feature-based View Reconstructions for Light Fields},
    author={Jinglei Shi and Xiaoran Jiang and Christine Guillemot},
    booktitle={IEEE. International Conference on Computer Vision and Pattern Recognition (CVPR)},
    pages={2555--2564},
    year={2020}}
```
