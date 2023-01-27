## Purpose
By using sequential visits (historical data) of a patient, we want to develop models that can predict whether or not a patient will develop AMD. For each of the historical data from 3, 6, 9, 12, 15, 18, 21 months, a model is trained using both LSTM and Transformers.

## Data Preprocessing
The data is initially indexed with fold numbers from 1 to 10. We used a five-fold CV. To convert 10 to 5, folds with the same raminder by 5 are put in the same fold.
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
half pyramid, means that we would have:0000x000xx00xxx0xxxxxxxxx

Another technique that was used to replicate better the test set is dropping (convert all of them to zero) a percentage of the rows after augmentation. In any case the last row or the row containing all the visits of the patient would not be dropped. So if percentage is 20%, except the last row, others have probability of 20% to become all zeros.

In the example above the length of each data is 5, however for each month prediction, we first find the max number of visists among all the patients and consider it as the global length for that prediction task. 
The validation data is only padded to be consistent.
## Loss function
Two options were considered to be tested as a loss functions
1- weighted categorical cross-entropy
2- focal loss

## Optimizers
1- Adam
2- RMSprop

## LSTM Layer Unit numbers
different number of units are tested for the two LSTM layers: 5, 10, 20, 25, 30, 50

## Training Results
For each prediction task (e.g. 3month, 5 month, etc), we would have 30 models (5 folds and 6 different LSTM models). Obviously, by adding the othercombinations such as loss function, droping percentage, this number increase a lot. After each training, the model weights of that particular task (3 months, 5 months, andetc), with the corresponding LSTM unit's number, fold's number, ground truth and prediction results over the validation set, and auc are stored. For instance:OCT_model_with_weights_12_25_1.h5
corresponds to the model that is trained over the 12 months data, the LSTM layers units is 25 and the validation set was 1 whereas 2, 3, 4, 5 were used as the training. 

## Testing
In the test session, the prediction and ground truth of five folds are used to create the roc curve and obtain auc. Since we have 6 different number of layers, we would have 6 distinct auc values: 1- auc of 5-folds when LSTM layers have 5 units2- auc of 5-folds when LSTM layers have 10 units3- auc of 5-folds when LSTM layers have 20 units4- auc of 5-folds when LSTM layers have 25 units5- auc of 5-folds when LSTM layers have 30 units6- auc of 5-folds when LSTM layers have 50 units
Accordingly, the highest auc is chosen among the top 6. In this way, we can understand what is the best value for the LSTMs' units. Then again, for each of them we have 5 different models and we choose the one with the highest auc. 
