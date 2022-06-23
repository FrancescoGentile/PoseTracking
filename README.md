# Installation

python 3.7.13

pytorch 1.11.0

CUDA 11.3

## Setup

```
conda create --name tracking python=3.7 dask -y
conda activate tracking
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch -y
pip install gdown

git clone https://github.com/FrancescoGentile/PoseTracking.git
cd PoseTracking 
pip install -r requirements.txt

cd thirdparty/deform_conv
python setup.py develop

cd ../../
python setup.py develop
pip install cython
pip install 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
pip install cython_bbox
```

## Download models 

```
cd PoseTracking

mkdir pretrained
cd pretrained 
gdown 1P4mY0Yyd3PPTybgZkjMYhFri88nTmJX5

cd ../
mkdir -p DcPose_supp_files/pretrained_models/DCPose
cd DcPose_supp_files/pretrained_models/DCPose
gdown 1u4SGKOcR2YuXjBwGOxrGcmSu5BjrqsD3
mv PoseTrack17_DCPose.pth.pth\（副本）   PoseTrack17_DCPose.pth

cd ../../../
mkdir -p DcPose_supp_files/object_detector/YOLOv3
cd DcPose_supp_files/object_detector/YOLOv3
gdown 11cLTVEUj7rXNPsnuzMaqzFOC1kCEH5oC
```

# Inference

```
cd PoseTracking/tracking 

python main.py \
    --input ./input \
    --labels ./labels.json \
    --output ./output \
    --batch 10 \
    --root-dir ../ \
    -f ../exps/example/mot/yolox_x_mix_det.py \
    -c ../pretrained/bytetrack_x_mot17.pth.tar \
```