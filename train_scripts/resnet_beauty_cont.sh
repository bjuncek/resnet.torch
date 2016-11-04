th main.lua \
-depth 34 \
-batchSize 32 \
-nGPU 1 \
-nThreads 4 \
-data /media/eightbit/data_hdd/Projects/BeautyOrNot/TorchData/ \
-dataset beauty \
-nClasses 10 \
-resetClassifier true \
-backend cudnn \
-nEpochs 2 \
-retrain /media/eightbit/data_hdd/Libs/fb.resnet.torch/my_models/trained_models_beauty/model_28.t7 \
-LR 0.00005 \
-LR_decay_step 10 \
-gen cache_files  \
-save my_models/trained_models_beauty/ \
-checkpoint true \