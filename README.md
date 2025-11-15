# IGEV-StereoLite
Research Source: 

Real-time stereo reconstruction and geometric quantification of pavement distress with a variable-baseline platform

# Iterative Geometry Encoding Volume stereo-Lite
> Core code reference: igev_lite and associated files in the core directory

## 1. Environment Setup
Install other dependencies
```
pip install -r requirements.txt
```
## 2. Evaluation Datasets
```data
└── Datasets
    |- kitti12/15 (kitti.yaml)
    |- SceneFlow (sceneflow.yaml)
    |- Sintel (sintel.yaml)
    |- RSRD (RSRD.yaml)
```

## 3. Testing
```
python testModel.py
```

## 4. Network
> IGEV-Stereo
![img.png](IGEV-Stereo.png)
> IGEV-Stereo Lite
![img.png](IGEV-StereoLite.png)
> Demo
![img.png](demo-imgs.png)
