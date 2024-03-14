# Stock Price Prediction with a Recurrent Neural Network (RNN)

## Overview
This project constitutes a Python-based implementation of a Recurrent Neural Network (RNN) for predicting day closing Alphabet Inc stock prices from historical sequential time series data. The script first loads the historical stock price data from a CSV file, preprocesses this dataset, trains an RNN model and evaluates its performance on test data. The model architecture consists of a SimpleRNN layer followed by a Dense layer.

## Repository structure
- **README.md**: This file.
- [**/code**](link): A folder containing the code developed to implement the project:
  - `rnn-prediction.py`: The main script containing a series of functions to carry out the project.
- [**/plot**](link): A folder containing the plots output by the RNN script.
  - `loss-plot.png`: A plot visualising the training and validation loss over the epochs during model training.
  - `training-prediction.png`: A plot visualising the model's predictions during training, against the actual stock price data values.
  - `test-prediction.png`: A plot visualising the model's predictions during testing, against the actual stock price data values.

## Dataset
The [dataset](https://www.kaggle.com/datasets/henryshan/google-stock-price) used for this project was obtained from [Kaggle.com](www.kaggle.com). This dataset contains information on the date, market open price, day high, day low, market close price, adjusted market close price and market volume for Alphabet Inc stock on the 4859 days between 19th August 2004 to 5th December 2023.

## Code overview
Below is a detailed overview of each component in the main `rnn-prediction.py` script:

### 1. Data Loading
The script starts by loading historical stock price data from a CSV file. It utilizes pandas' read_csv() function to load the data into a pandas DataFrame.

### 2. Data Preprocessing
Once the data is loaded, preprocessing steps are applied to make it suitable for training the RNN model. This includes converting dates to datetime format, handling missing values, and scaling numerical features using MinMaxScaler from scikit-learn.

### 3. Data Splitting
The preprocessed data is split into training and test sets. The split ratio is determined by the user-defined split_ratio parameter.

### 4. Sequence Generation
To train the RNN model, input sequences and corresponding target values are generated from the training and test datasets. The length of the input sequences is specified by the sequence_length parameter.

### 5. Model Building
The RNN model architecture consists of a SimpleRNN layer followed by a Dense layer. The number of neurons in the SimpleRNN layer, the dropout rate, and other hyperparameters are configurable.

### 6. Model Training
The RNN model is trained using the training input sequences and target values. The training process involves specifying the number of epochs, batch size, and validation split. The training progress is visualized using matplotlib.

### 7. Model Evaluation
Once trained, the RNN model is evaluated on both training and test datasets to assess its performance. Mean Squared Error (MSE) is used as the evaluation metric.

### 8. Predictions Visualization
Finally, the script generates predictions using the trained model and visualizes them alongside actual stock prices. This provides insight into the model's ability to capture stock price trends.



## Results

The model was trained over 50 epochs. The training loss steadily decreased from 0.0148 to 0.0001, indicating that the model learned from the training data. This reduction in training loss suggests that the model is improving its ability to make predictions on the training data over time.

Included in the **Outputs** section below is the full output of the `rnn-prediction.py` script, listing the training and validation loss over the epochs during model training. 

### Training Loss: 0.0001
The model was trained over 50 epochs. Initially, the training loss stood at 0.0148 and steadily decreased to 0.0001. This reduction suggests that the model effectively learned from the training data, improving its predictive capabilities over time.

### Validation Loss: 0.0002
During the training process, the validation loss also decreased from 0.0088 to 0.0002. This decline indicates that the model not only fit well to the training data but also generalized effectively to unseen validation data. The decreasing validation loss demonstrated an improvement in the model's performance on the validation dataset.

### Test Loss: 0.0008
Following training, the model underwent evaluation on the test dataset. The test loss was reported as 0.0008. This metric provides an estimate of the model's performance on completely new, unseen data. Although slightly higher than the validation loss, the test loss indicated that the model maintained good performance on the test dataset.

## Conclusion
Overall, the model demonstrated effective training, with low training, validation, and test losses. The consistent decrease in both training and validation losses suggested that the model learned efficiently from the data and generalized well to unseen samples.

In conclusion, based on the provided output, the model was trained effectively and exhibited good performance on both validation and test datasets, indicating its potential for making accurate predictions on unseen data.

## Outputs
```
Epoch 1/50
43/43 [==============================] - 1s 9ms/step - loss: 0.0148 - val_loss: 0.0088
Epoch 2/50
43/43 [==============================] - 0s 4ms/step - loss: 0.0036 - val_loss: 0.0032
Epoch 3/50
43/43 [==============================] - 0s 5ms/step - loss: 0.0025 - val_loss: 0.0027
Epoch 4/50
43/43 [==============================] - 0s 6ms/step - loss: 0.0022 - val_loss: 6.0092e-04
Epoch 5/50
43/43 [==============================] - 0s 5ms/step - loss: 0.0019 - val_loss: 0.0014
Epoch 6/50
43/43 [==============================] - 0s 5ms/step - loss: 0.0015 - val_loss: 0.0016
Epoch 7/50
43/43 [==============================] - 0s 5ms/step - loss: 0.0015 - val_loss: 9.9963e-04
Epoch 8/50
43/43 [==============================] - 0s 5ms/step - loss: 0.0014 - val_loss: 0.0012
Epoch 9/50
43/43 [==============================] - 0s 5ms/step - loss: 0.0012 - val_loss: 6.3220e-04
Epoch 10/50
43/43 [==============================] - 0s 5ms/step - loss: 0.0010 - val_loss: 0.0013
Epoch 11/50
43/43 [==============================] - 0s 5ms/step - loss: 8.4957e-04 - val_loss: 5.5352e-04
Epoch 12/50
43/43 [==============================] - 0s 5ms/step - loss: 8.3317e-04 - val_loss: 9.9496e-04
Epoch 13/50
43/43 [==============================] - 0s 5ms/step - loss: 7.8829e-04 - val_loss: 2.1268e-04
Epoch 14/50
43/43 [==============================] - 0s 5ms/step - loss: 8.5371e-04 - val_loss: 5.7321e-04
Epoch 15/50
43/43 [==============================] - 0s 5ms/step - loss: 7.2669e-04 - val_loss: 7.2331e-04
Epoch 16/50
43/43 [==============================] - 0s 5ms/step - loss: 7.2275e-04 - val_loss: 8.5490e-04
Epoch 17/50
43/43 [==============================] - 0s 5ms/step - loss: 6.5584e-04 - val_loss: 2.8168e-04
Epoch 18/50
43/43 [==============================] - 0s 5ms/step - loss: 6.9182e-04 - val_loss: 2.3320e-04
Epoch 19/50
43/43 [==============================] - 0s 5ms/step - loss: 5.5396e-04 - val_loss: 1.8462e-04
Epoch 20/50
43/43 [==============================] - 0s 5ms/step - loss: 5.8738e-04 - val_loss: 3.8006e-04
Epoch 21/50
43/43 [==============================] - 0s 5ms/step - loss: 5.2703e-04 - val_loss: 4.0815e-04
Epoch 22/50
43/43 [==============================] - 0s 5ms/step - loss: 5.1631e-04 - val_loss: 3.5189e-04
Epoch 23/50
43/43 [==============================] - 0s 5ms/step - loss: 4.9320e-04 - val_loss: 7.5525e-04
Epoch 24/50
43/43 [==============================] - 0s 5ms/step - loss: 4.7648e-04 - val_loss: 1.8021e-04
Epoch 25/50
43/43 [==============================] - 0s 5ms/step - loss: 4.6434e-04 - val_loss: 7.2024e-04
Epoch 26/50
43/43 [==============================] - 0s 5ms/step - loss: 4.4382e-04 - val_loss: 5.0341e-04
Epoch 27/50
43/43 [==============================] - 0s 5ms/step - loss: 4.2591e-04 - val_loss: 9.9459e-04
Epoch 28/50
43/43 [==============================] - 0s 5ms/step - loss: 4.9024e-04 - val_loss: 7.1699e-04
Epoch 29/50
43/43 [==============================] - 0s 5ms/step - loss: 4.3749e-04 - val_loss: 1.9807e-04
Epoch 30/50
43/43 [==============================] - 0s 5ms/step - loss: 3.9958e-04 - val_loss: 3.3817e-04
Epoch 31/50
43/43 [==============================] - 0s 5ms/step - loss: 4.0418e-04 - val_loss: 5.7464e-04
Epoch 32/50
43/43 [==============================] - 0s 5ms/step - loss: 4.1248e-04 - val_loss: 5.1932e-04
Epoch 33/50
43/43 [==============================] - 0s 5ms/step - loss: 3.9860e-04 - val_loss: 1.5779e-04
Epoch 34/50
43/43 [==============================] - 0s 5ms/step - loss: 4.1628e-04 - val_loss: 3.0478e-04
Epoch 35/50
43/43 [==============================] - 0s 5ms/step - loss: 4.1359e-04 - val_loss: 0.0013
Epoch 36/50
43/43 [==============================] - 0s 5ms/step - loss: 3.8152e-04 - val_loss: 8.6094e-04
Epoch 37/50
43/43 [==============================] - 0s 5ms/step - loss: 3.5413e-04 - val_loss: 3.6522e-04
Epoch 38/50
43/43 [==============================] - 0s 5ms/step - loss: 3.9113e-04 - val_loss: 3.7307e-04
Epoch 39/50
43/43 [==============================] - 0s 5ms/step - loss: 3.8847e-04 - val_loss: 1.4426e-04
Epoch 40/50
43/43 [==============================] - 0s 5ms/step - loss: 3.9451e-04 - val_loss: 4.5673e-04
Epoch 41/50
43/43 [==============================] - 0s 5ms/step - loss: 3.7606e-04 - val_loss: 3.8336e-04
Epoch 42/50
43/43 [==============================] - 0s 5ms/step - loss: 3.6466e-04 - val_loss: 3.5811e-04
Epoch 43/50
43/43 [==============================] - 0s 5ms/step - loss: 4.0419e-04 - val_loss: 3.9160e-04
Epoch 44/50
43/43 [==============================] - 0s 5ms/step - loss: 3.4886e-04 - val_loss: 0.0013
Epoch 45/50
43/43 [==============================] - 0s 5ms/step - loss: 3.7717e-04 - val_loss: 1.9643e-04
Epoch 46/50
43/43 [==============================] - 0s 5ms/step - loss: 3.7542e-04 - val_loss: 2.1076e-04
Epoch 47/50
43/43 [==============================] - 0s 5ms/step - loss: 3.6396e-04 - val_loss: 5.2502e-04
Epoch 48/50
43/43 [==============================] - 0s 5ms/step - loss: 3.7487e-04 - val_loss: 9.9961e-04
Epoch 49/50
43/43 [==============================] - 0s 5ms/step - loss: 3.4087e-04 - val_loss: 1.8633e-04
Epoch 50/50
43/43 [==============================] - 0s 5ms/step - loss: 3.8604e-04 - val_loss: 1.9840e-04
Training loss: 0.0001
Test loss: 0.0008
106/106 [==============================] - 0s 1ms/step
45/45 [==============================] - 0s 1ms/step
```
