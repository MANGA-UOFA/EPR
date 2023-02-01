# Explaining Natural Language Inference with EPR

## 1. Setup enviroment 
```bash
pip install -r requirements.txt
```

## 2. Run experiment with experiment script
The script feature the inputs for running the experiment
```bash
bash ./scripts/experiment.sh $SAVE_LOC $DATA_LOC $EPOCHS $CEPOCHS $BATCH_SIZE $LR $CLR
```
To reproduce our paper result, we recommend 
```bash
bash ./scripts/experiment.sh $SAVE_LOC data 10 10 32 3e-4 3e-6
```
## 3. Run test with test script
The script provides feature the inputs for running the testing
```bash
bash ./scripts/test.sh $SAVE_LOC $DATA_LOC $EVAL_BATCH $RUN_CKPT
```
Evaluate the best checkpoint by changing the $RUN_CKPT
```bash
bash ./scripts/test.sh $SAVE_LOC data $EVAL_BATCH $RUN_CKPT
```
