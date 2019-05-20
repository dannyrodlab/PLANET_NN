# PLANET_NN
## Neural Networks to solve the Kaggle Challenge.

### How to use this code
Your home folder should like:
```
HOME
└───data
│   │   train2.csv
│   └───fast_val
│   └───fast_train
└───PLANET_NN
│   │   README.ME 
└───scripts    
└───results
└───checkpoints    
```
Then, you must run the code as follows:
```console
export PATH=/home/administrador/anaconda3/bin:$PATH
screen -S 201905200121_BATMAN_ALEXNET_BATCH_50_EPOCH_30
python scripts/train_NN.py --name "201905200120_resnet_batch_50_epoch_30" --model "resnet" --batch 50 --epoch 30
```
Alternatively, you can globally export the PATH. I did this once, but was unable to replicate it.
### How to name my experiments
Even though it seems stupidly complex, please pass name as 'year-month-day-hour-minute_model_batch_SIZE_epoch_NUMBER'. It makes it a lot easier for the rest.
