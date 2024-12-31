# Conv2Former

The official implementation of the paper "[Conv2Former: A Simple Transformer-Style ConvNet for Visual Recognition](https://arxiv.org/abs/2211.11943)". Our code is based on [timm](https://github.com/rwightman/pytorch-image-models) and [ConvNeXt](https://github.com/facebookresearch/ConvNeXt).

Our paper is accepted by IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI).

## Usage

### Requirements

- torch>=1.12.1
- torchvision>=0.13.1
- timm>=0.9.12

### Examples

```python
from models.conv2former import conv2former_n

# create a Conv2Former model with 1000 classes
model = conv2former_n(num_classes=1000) 
``` 

Input image should be normalized as follows:

```python
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225])
``` 

We also provide the [Jittor](https://cg.cs.tsinghua.edu.cn/jittor/) version of the model:

```python
import jittor as jt
from jittor_models.conv2former import conv2former_n
   
# create a Conv2Former model with 1000 classes
model = conv2former_n(num_classes=1000) 
``` 

### Training

```bash
bash distributed_train.sh 8 $DATA_DIR --model $MODEL -b $BS --lr $LR --drop-path $DROP_PATH

# DATA_DIR: path to the dataset
# MODEL: name of the model
# BS: batch size
# LR: learning rate
# DROP_PATH: drop path rate
```

### validation

```bash
python validate.py $DATA_DIR --model $MODEL --checkpoint $CHECKPOINT

# DATA_DIR: path to the dataset
# MODEL: name of the model
# CHECKPOINT: path to the saved checkpoint
```

## Results


### Training on ImageNet-1k

| Model                | Parameters | FLOPs  | Image resolution | Top 1 Acc. | Model File   |
| :------------------- | :--------- | :------| :--------------- | :--------- | :----------- |
| Conv2Former-N        | 15M        | 2.2G   |       224        |  81.5%     | [baiduyun](https://pan.baidu.com/s/1sgO-UhF27UQrGtfKJzck3A?pwd=6ku9)      |
| SwinT-T              | 28M        | 4.5G   |       224        |  81.5%     | -            |
| ConvNeXt-T           | 29M        | 4.5G   |       224        |  82.1%     | -            |
| Conv2Former-T        | 27M        | 4.4G   |       224        |  83.2%     | [baiduyun](https://pan.baidu.com/s/1Rfsi5p5usr3UE46jigtlcg?pwd=gwq4)      |
| SwinT-S              | 50M        | 8.7G   |       224        |  83.0%     | -            |
| ConvNeXt-S           | 50M        | 8.7G   |       224        |  83.1%     | -            |
| Conv2Former-S        | 50M        | 8.7G   |       224        |  84.1%     | [baiduyun](https://pan.baidu.com/s/16Lu1LcQkOoSQXnZVGxGPxg?pwd=6z6m)      |
| RepLKNet-31B         | 79M        | 15.3G  |       224        |  83.5%     | -            |
| SwinT-B              | 88M        | 15.4G  |       224        |  83.5%     | -            |
| ConvNeXt-B           | 89M        | 15.4G  |       224        |  83.8%     | -            |
| FocalNet-B           | 89M        | 15.4G  |       224        |  83.9%     | -            |
| Conv2Former-B        | 90M        | 15.9G  |       224        |  84.4%     | [baiduyun](https://pan.baidu.com/s/1q5V1J3jzAoKQis-hx6rTVg?pwd=bdgj)      |

### Pre-Training on ImageNet-22k and Finetining on ImageNet-1k

| Model                | Parameters | FLOPs  | Image resolution | Top 1 Acc. | Model File   |
| :------------------- | :--------- | :------| :--------------- | :--------- | :----------- |
| ConvNeXt-S           | 50M        | 8.7G   |       224        |  84.6%     |  -           |
| Conv2Former-S        | 50M        | 8.7G   |       224        |  84.9%     | [baiduyun](https://pan.baidu.com/s/17dq0ScQ8e5wOqZHfu6nfDw?pwd=wg3s)      |
| SwinT-B              | 88M        | 15.4G  |       224        |  85.2%     | -            |
| ConvNeXt-B           | 89M        | 15.4G  |       224        |  85.8%     | -            |
| Conv2Former-B        | 90M        | 15.9G  |       224        |  86.2%     | [baiduyun](https://pan.baidu.com/s/1TlIacEV4xaedQRKoL5wxTQ?pwd=emdg)      |
| SwinT-B              | 88M        | 47.0G  |       384        |  86.4%     | -            |
| ConvNeXt-B           | 89M        | 45.1G  |       384        |  86.8%     | -            |
| Conv2Former-B        | 90M        | 46.7G  |       384        |  87.0%     | -      |
| SwinT-L              | 197M       | 34.5G  |       224        |  86.3%     | -            |
| ConvNeXt-L           | 198M       | 34.4G  |       224        |  86.6%     | -            |
| Conv2Former-L        | 199M       | 36.0G  |       224        |  87.0%     | [baiduyun](https://pan.baidu.com/s/1GNKsMsO4oStGqKAwV8eltQ?pwd=dyfj)      |
| EffNet-V2-XL         | 208M       | 94G    |       480        |  87.3%     | -            |
| SwinT-L              | 197M       | 104G   |       384        |  87.3%     | -            |
| ConvNeXt-L           | 198M       | 101G   |       384        |  87.5%     | -            |
| CoAtNet-3            | 168M       | 107G   |       384        |  87.6%     | -            |
| Conv2Former-L        | 199M       | 106G   |       384        |  87.7%     | [baiduyun](https://pan.baidu.com/s/1kXUvZLRpIA8i_vZ-HroTZw?pwd=iwm9)      |


### Citation

If you find this work or code is helpful in your research, please cite:
```
@article{hou2024conv2former,
  title={Conv2Former: A Simple Transformer-Style ConvNet for Visual Recognition},
  author={Hou, Qibin and Lu, Cheng-Ze and Cheng, Ming-Ming and Feng, Jiashi},
  journal={IEEE TPAMI},
  year={2024},
  doi={10.1109/TPAMI.2024.3401450}, 
}
```

### Reference
You may want to cite:
```
@inproceedings{liu2022convnet,
      title={A ConvNet for the 2020s}, 
      author={Zhuang Liu and Hanzi Mao and Chao-Yuan Wu and Christoph Feichtenhofer and Trevor Darrell and Saining Xie},
      booktitle=CVPR,
      year={2022}
}

@inproceedings{liu2021swin,
  title={Swin transformer: Hierarchical vision transformer using shifted windows},
  author={Liu, Ze and Lin, Yutong and Cao, Yue and Hu, Han and Wei, Yixuan and Zhang, Zheng and Lin, Stephen and Guo, Baining},
  booktitle=ICCV,
  year={2021}
}

@inproceedings{tan2021efficientnetv2,
  title={Efficientnetv2: Smaller models and faster training},
  author={Tan, Mingxing and Le, Quoc},
  booktitle=ICML,
  pages={10096--10106},
  year={2021},
  organization={PMLR}
}

@misc{focalmnet,
  author = {Yang, Jianwei and Li, Chunyuan and Gao, Jianfeng},
  title = {Focal Modulation Networks},
  publisher = {arXiv},
  year = {2022},
}

@article{dai2021coatnet,
  title={Coatnet: Marrying convolution and attention for all data sizes},
  author={Dai, Zihang and Liu, Hanxiao and Le, Quoc and Tan, Mingxing},
  journal=NIPS,
  volume={34},
  year={2021}
}

@inproceedings{replknet,
  author = {Ding, Xiaohan and Zhang, Xiangyu and Zhou, Yizhuang and Han, Jungong and Ding, Guiguang and Sun, Jian},
  title = {Scaling Up Your Kernels to 31x31: Revisiting Large Kernel Design in CNNs},
  booktitle=CVPR,
  year = {2022},
}
```