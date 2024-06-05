# KMLFC
This is the repository for paper "**Learning Kernel-Modulated Neural Representation for Efficient Light Field Compression**" (**arXiv-2307.06143**).

By [Jinglei Shi](https://jingleishi.github.io/),  [Yihong Xu](https://github.com/yihongXU)  and  [Christine Guillemot](https://people.rennes.inria.fr/Christine.Guillemot/)

<[Paper link](https://arxiv.org/abs/2307.06143)>

## Dependencies
```
python=3.8.18
numpy=1.24.3
matplotlib=3.7.2
hydra-core==1.3.2
omegaconf==2.3.0
pytorch=1.11.0
torchvision==0.12.0
pytorch-lightning=2.0.9
scipy=1.10.1
```

## Training
Coming soon

## Download Pretrained Models
We provide pretrained models for four synthetic HCI scenes (*hci.zip*), four realworld scenes captured using Lytro camera (*lytro.zip*), and three challenging scenes (*challenging.zip*). Users can download them via [PanBaidu](https://pan.baidu.com/s/1no2sBxrRyax97JPB5F4aHQ?pwd=lfcc) or [Google Drive](https://drive.google.com/drive/folders/16ZU0tn7sn0hQOkqJMLN8GowCsmjGd2SZ?usp=sharing), then create a new folder named 'results' and put the unzipped files into it. 

We offer seven models for each scene, including ($c_m$, $c_d$)={(2,48),(2,63),(2,78),(2,93),(2,123),(2,153),(2,183)}, where each model corresponds to a specific architecture setting. For instance, '*checkpoints_c50_a1*' indicates that the number of channels for the modulator $c_m/2$ in each angular direction is 1, and the total number of channels for both modulator and descriptor is 50 (the number of channels for the descriptor $c_d$ is calculated as 50-2*1=48).

## Evaluation
To launch the generation of the light fields (decoding), users should first configure the file '***batch_infer.py***' as follows:
- **Lines 41-43**: Specify the target light fields to be generated along with their spatial resolution. For example: scene_list = [['boxes', 512, 512]].
- **Lines 45-47**: Define the target network architecture. For example: architecture_list = [[[50]*5, [1]*5]], this setting corresponds to the checkpoint 'checkpoints_c50_a1'.

After configuring the above settings, users can simply launch the simulation by executing:
```
python batch_infer.py
```
The generated light fields will be accessible in the folder 'results/lf_name/test_cxx_axx'.

## Related Projects
Another two projects related to this work will be released soon, they are:

QDLR-NeRF (ICASSP-2023)

DDLF (TIP-2023)

Feel free to use and cite them!

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
