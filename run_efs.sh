rm -r build
cd detectron2
python setup.py build develop
cd ../

export DETECTRON2_DATASETS=/data/

# ============================= Dual Decision (K% classes are set as unknown) ==========================
# K=5
python train_net.py --dist-url tcp://127.0.0.1:21032 --config-file configs/ours_K5.yaml --num-gpus 8 DATALOADER.NUM_WORKERS 16

# K=10
python train_net.py --dist-url tcp://127.0.0.1:21032 --config-file configs/ours_K10.yaml --num-gpus 8 DATALOADER.NUM_WORKERS 16

# K=20
python train_net.py --dist-url tcp://127.0.0.1:21032 --config-file configs/ours_K20.yaml --num-gpus 8 DATALOADER.NUM_WORKERS 16