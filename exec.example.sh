python process.py \
    --input ./input \
    --labels /path/to/kinetics400/test.json \
    --output ./output \
    -f exps/example/mot/yolox_x_mix_det.py \
    -c pretrained/bytetrack_x_mot17.pth.tar \
    --fp16 --fuse