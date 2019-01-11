## Speed-challenge

Comma.ai speed challenge attempt.
Goal: Predict the speed of a driving car from a dashcam.
I used a CNN which input is a Farnback optical flow between two frames, regression goal was the average of the speed of these two frames

### Run:
**Prepare data**
```bash
python data_prep.py
```
**Train the network**
 - split: Validation/Training split
 - data/prepared-data/train_flow_meta.csv: default location of the CSV file mapping optical flow path and averaged speed
```bash
python train.py --split 0.3 data/prepared-data/train_flow_meta.csv
```
**Analyze the loss and predict the speed of the test video's frames**

Open the _"Analysis & Prediction"_ notebook

**Comment** 

I got an MSE of 1 on the training video.
After overlaying the predicted "speed" (because it's neither miles/h or km/h) on the test video, the result look quite convincing.

The only part of the test video where the model struggles is at the 00:57 timestamp where the car is stopped and cars are driving perpendicular to it. There were no examples of this in the training set and the driving cars confuses the prediction