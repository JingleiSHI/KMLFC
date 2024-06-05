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

### Download Pretrained Models
We provide pretrained models of four synthetic HCI scenes ([hci.zip](https://pan.baidu.com/s/1NJcAUftGhHtmudTvAXt6Wg?pwd=lfcc)), of three realworld scenes captured using Lytro camera ([lytro.zip](https://pan.baidu.com/s/1GIC4zpFC9mOgCl0wL1qQIQ?pwd=lfcc)), and of three challenging scenes ([challenging.zip](https://pan.baidu.com/s/1SbiNtcGcr3cwqBhHVnF4Jw?pwd=lfcc))

### Evaluation

## Citation
Please consider citing our work if you find it useful.
```
@article{shi2023learning,
  title={Learning Kernel-Modulated Neural Representation for Efficient Light Field Compression},
  author={Jinglei Shi and Yihong Xu and Christine Guillemot},
  journal={arXiv preprint arXiv:2307.06143},
  year={2023}
}
```
