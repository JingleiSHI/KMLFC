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

### Training
Coming soon

### Download Pretrained Models
We provide pretrained models for four synthetic HCI scenes (*hci.zip*), four realworld scenes captured using Lytro camera (*lytro.zip*), and three challenging scenes (*challenging.zip*). Users can download them via [PanBaidu](https://pan.baidu.com/s/1no2sBxrRyax97JPB5F4aHQ?pwd=lfcc) or [Google Drive](https://drive.google.com/drive/folders/16ZU0tn7sn0hQOkqJMLN8GowCsmjGd2SZ?usp=sharing), then unzip them and put them into the created folder 'results'. 

We offered 7 models for every scene, with each model corresponding to a specific architecture setting, for instance, '*checkpoints_c50_a1*' means the number of channel for modulator in each angular direction is 1, and the total number of channel for both modulator and descriptor is 50 (the number of channel for descriptor is 50-2*1=48).

### Evaluation
To launch the generation of the light fields (decoding), users should first configure the file 'batch_infer.py' as follows:
- Line 16-18: The path to the folders 'results', 'data', 'input_noise', where the folder 'results' is used to store the generated light fields, the folder 'data' is used to store the light fields to be learned during the training (which is not used for inference) and the folder 'input_noise' 

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
