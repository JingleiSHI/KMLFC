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
We provide pretrained models for four synthetic HCI scenes (hci.zip), three realworld scenes captured using Lytro camera (lytro.zip), and three challenging scenes (challenging.zip). Users can download them via [PanBaidu](https://pan.baidu.com/s/1no2sBxrRyax97JPB5F4aHQ?pwd=lfcc) or [Google Drive](https://drive.google.com/drive/folders/16ZU0tn7sn0hQOkqJMLN8GowCsmjGd2SZ?usp=sharing), then unzip them and put them into the folder 'results'

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
