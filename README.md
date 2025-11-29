# Low-Rank Head Avatar Personalization with Registers (NeurIPS 2025)
Code for [Low-Rank Head Avatar Personalization with Registers](https://openreview.net/pdf?id=mhARf5VzCn "https://openreview.net"). Also available on [arxiv](https://arxiv.org/abs/2506.01935). [Project Page](https://starc52.github.io/publications/2025-05-28-LoRAvatar/)
## Usage
```
cd LoRAvatar
```

### Build environment (from GAGAvatar)
```
conda env create -f environment.yml
conda activate GAGAvatar
```

### Install 3DGS
```
git clone --recurse-submodules git@github.com:xg-chu/diff-gaussian-rasterization.git
pip install ./diff-gaussian-rasterization
rm -rf ./diff-gaussian-rasterization
```

### Prepare Resources
```
bash ./build_resources.sh
```

### Prepare GAGAvatar_track resources
```
cd core/libs/GAGAvatar_track
bash ./build_resources.sh
```

Install ```minLoRA``` like below in gagavatar environment. If directory minLoRA already exists, skip the ```git clone``` step. 
```
git clone https://github.com/cccntu/minLoRA.git
cd minLoRA
pip install -e .
```
Install additional dependencies for gaga from ```additional_requirements_for_gaga.txt``` in gagavatar environment. 

Install additional dependencies for gaga track from ```additional_requirements_for_gaga_track.txt``` in gagavatar_track environment.

## Quick Start Guide
In order to use our method, some additional preprocessing is required. Use these commands to get the following.  Uses a changed ```track_video.py```.
```
cd core/libs/GAGAvatar_track/
python track_video.py -v /path/to/identity/video.mp4 # tracks video according to GAGAvatar track. Also computes visible flame vertices, necessary for our register module. 
python revise_visible_vertices.py -p /path/to/tracked/folder/smoothed.pkl # computes projection points, mask points, topk neighbours for mask points, necessary for our register module. 
```
To adapt to a identity with our register module use
```
python lora_adaptation.py -i /path/to/source_img.png -d /path/to/tracked_video/ --threed_register --threed_register_loss
```
To infer with LoRA weights trained with our register module use
```
python lora_inference.py -i /path/to/source_img.png -d /path/to/driving/video -t <tracked_identity_name>  -l <tracked_identity_name>  --threed_register --threed_register_loss
```
Here, ```<tracked_identity_name>``` should be the output of ```os.path.basename(/path/to/tracked/video)``` used during adaptation. 


To adapt to a identity with our register module and DoRA use
```
python dora_adaptation.py -i /path/to/source_img.png -d /path/to/tracked_video/ --threed_register --threed_register_loss
```
To infer with DoRA weights trained with our register module use
```
python dora_inference.py -i /path/to/source_img.png -d /path/to/driving/video -t <tracked_identity_name>  -l <tracked_identity_name>  --threed_register --threed_register_loss
```


## Acknowledgements
This code has been adapted from and built on top of GAGAvatar and GAGAvatar_track. We also thank the following projects for sharing their great work!
* GPAvatar: https://github.com/xg-chu/GPAvatar
* FLAME: https://flame.is.tue.mpg.de
* StyleMatte: https://github.com/chroneus/stylematte
* EMICA: https://github.com/radekd91/inferno
* VGGHead: https://github.com/KupynOrest/head_detector


If you use this work, please be so kind to cite us: 

```
@inproceedings{
chakkera2025lowrank,
title={Low-Rank Head Avatar Personalization with Registers},
author={Sai Tanmay Reddy Chakkera and Aggelina Chatziagapi and Md Moniruzzaman and Chen-ping Yu and Yi-Hsuan Tsai and Dimitris Samaras},
booktitle={The Thirty-ninth Annual Conference on Neural Information Processing Systems},
year={2025},
url={(https://openreview.net/pdf?id=mhARf5VzCn)}
}
```
