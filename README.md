# DL_Particle_Detection
Deep learning methods for particle detection in microscopy images.

It covers four deep-learning model, including DetNet[1], deepBlink[2], superpoint[3], and HigherHRNet[4]. HigherHRNet[4] was modified and renamed as PointDet in this repository.

## Environment
The code is developed using python 3.7.3 on Ubuntu 18.04. NVIDIA GPUs are needed. The code is developed and tested using 1 NVIDIA GeForce RTX 2080 Ti card.
## Quick Start
### Installation
1. Create a virtual environment
```
conda create -n dl_particle_detection python==3.7.3
conda activate dl_particle_detection
```
2. Install pytorch==1.8.0+cu111 torchvision==0.9.0+cu111
```
pip install torch==1.8.0+cu111 torchvision==0.9.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html
```
3. Clone this repo
```
git clone https://github.com/imzhangyd/DL_Particle_Detection.git
cd DL_Particle_Detection
```
4. Install dependencies:
 ```
 pip install -r requirements.txt
 ```

### Data preparation
Make your dataset look like this:
```
|-- data
`-- |-- dataset1
    `-- |-- train
        |   |-- image001.tif
        |   |-- image001.csv
        |   |-- image002.tif
        |   |-- image002.csv
        |   |-- ...
        `-- val
        |   |-- image001.tif
        |   |-- image001.csv
        |   |-- image002.tif
        |   |-- image002.csv
        |   |-- ...
        `-- test
        |   |-- image001.tif
        |   |-- image001.csv
        |   |-- image002.tif
        |   |-- image002.csv
        |   |-- ...
```
The .csv format:

x0,y0,0
x1,y1,0
x2,y2,0
...

The coordinate system is based on the left-top corner of the image, where the origin is located. The positive x-direction extends to the right, and the positive y-direction extends downward.

### Training
```
python trainval.py
```

### Testing
```
python infer_determine_thre.py
```


## Reference
[1] Wollmann, Thomas, et al. "DetNet: Deep neural network for particle detection in fluorescence microscopy images." 2019 IEEE 16th International Symposium on Biomedical Imaging (ISBI 2019). IEEE, 2019.  
[2] Eichenberger, Bastian Th, et al. "deepBlink: threshold-independent detection and localization of diffraction-limited spots." Nucleic Acids Research 49.13 (2021): 7292-7297.  
[3] DeTone, Daniel, Tomasz Malisiewicz, and Andrew Rabinovich. "Superpoint: Self-supervised interest point detection and description." Proceedings of the IEEE conference on computer vision and pattern recognition workshops. 2018.   
[4] Cheng, Bowen, et al. "Higherhrnet: Scale-aware representation learning for bottom-up human pose estimation." Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2020.
