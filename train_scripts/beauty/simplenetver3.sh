th main.lua \
-batchSize 48 \
-nGPU 1 \
-nThreads 5 \
-data /media/eightbit/data_ssd/0.5MARGIN/ \
-dataset beauty \
-nClasses 10 \
-backend cudnn \
-nEpochs 225 \
-classWeighting false \
-resetClassifier true \
-LR 0.05 \
-LR_decay_step 75 \
-gen cache_files/  \
-shareGradInput true \
-save my_models/beauty/ \
-netType simplenetver3 \
-model_init_LR -1 \
