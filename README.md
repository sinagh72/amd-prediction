## Purpose
By using sequential visits (historical data) of a patient, we want to develop models that can predict whether or not a patient will develop AMD. For each of the historical data from 3, 6, 9, 12, 15, 18, 21 months, a model is trained using both LSTM and Transformers.

## Data Preprocessing
The data is initially indexed with fold numbers from 1 to 10. We used a five-fold CV. To convert 10 to 5:
1. two consequtive folds are merged
2. folds with the same remainder of fold number by 5 are merged (remainder by 5)
### Padding
1. pre padding
2. post padding

### Augmentation
1. pyramid
2. half pyramid

Assuming that a patient has 5 visits (each denoted by x), the pyramid augmentation approach would be as follows:
+ 0000x
+ 000xx
+ 00xxx
+ 0xxxx
+ xxxxx
+ 0xxxx
+ 00xxx
+ 000xx
+ 0000x

In the case of post padding, x and 0 would be switched
half pyramid, means that we would have:
- 0000x
- 000xx
- 00xxx
- 0xxxx
- xxxxx

Another technique that was used to replicate better the test set is dropping (convert all of them to zero) a percentage of the rows after augmentation. In any case the last row or the row containing all the visits of the patient would not be dropped. So if percentage is 20%, except the last row, others have probability of 20% to become all zeros.

In the example above the length of each data is 5, however for each month prediction, we first find the max number of visists among all the patients and consider it as the global length for that prediction task. 
The validation data is only padded to be consistent.
## Loss function
Two options were considered to be tested as a loss functions
1. weighted categorical cross-entropy
2. focal loss

## Optimizers
1. Adam
2. RMSprop

## LSTM Layer Unit numbers
different number of units are tested for the two LSTM layers: 5, 10, 20, 25, 30, 50

## Training Results
For every prediction task (e.g. 3 months, 5 months, etc.), we would have 30 models (5 folds and 6 different LSTM models). This number increases a lot when other combinations are added, such as loss function and dropping percentage. A model weight for a particular task (3 months, 5 months, etc.) is stored, along with its corresponding LSTM unit number, fold number, and ground truth and prediction results over a validation set. As an example, OCT_model_with_weights_12_25_1.h5 represents the model trained over the 12 months of data. The LSTM layers are 25 and the validation set is 1, whereas the training set is 2, 3, 4, 5.
 

## Testing
A roc curve is created in the test session based on the prediction and ground truth of five folds. There would be 6 distinct auc values since we have 6 different layers: 
1. AUC of 5-folds when LSTM layers have 5 units
2. AUC of 5-folds when LSTM layers have 10 units
3. AUC of 5-folds when LSTM layers have 20 units
4. AUC of 5-folds when LSTM layers have 25 units
5. AUC of 5-folds when LSTM layers have 30 units
6. AUC of 5-folds when LSTM layers have 50 units

Among the top 6, the one with the highest AUC is selected. In this way, we can understand what is the best value for the LSTMs' units. Then again, for the selected LSTM unit number, the 5 models correspond to the 5-folds are compred and the one with the highest AUC over the validation is picked for inference.
