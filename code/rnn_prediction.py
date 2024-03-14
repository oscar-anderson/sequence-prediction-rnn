'''
Stock Price Prediction using Recurrent Neural Networks (RNN)

This script implements a Recurrent Neural Network (RNN) to predict stock price values. 
It loads historical stock price data from a CSV file, preprocesses it, trains an RNN model 
and evaluates its performance on test data. The model architecture includes a SimpleRNN layer 
followed by a Dense layer.

Author: Oscar Anderson
'''

# Import dependencies.
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense, Dropout
import matplotlib.pyplot as plt

# Load data.
def load_data(data_path: str) -> pd.DataFrame:
    '''
    Load data from a CSV file.

    Parameters:
        data_path (str): The path to the CSV file containing the data.

    Returns:
        pd.DataFrame: A DataFrame containing the loaded data.
    '''
    df = pd.read_csv(data_path)
    return df

# Split data.
def split_data(df: pd.DataFrame, split_ratio: float) -> tuple:
    '''
    Split data into training and test sets.

    Parameters:
        df (pd.DataFrame): The DataFrame containing the data to be split.
        split_ratio (float): The ratio of data to be allocated to the training set.
    
    Returns:
        tuple: A tuple containing two DataFrames: (training_df, test_df).
            - training_df (pd.DataFrame): The DataFrame containing the training data.
            - test_df (pd.DataFrame): The DataFrame containing the test data.
    '''
    num_data = len(df)
    print('Instances in all data: ', num_data)

    num_training = round(num_data * split_ratio)
    print('Instances in training data: ', num_training)
    training_df = df[:num_training]
    print(training_df)

    num_test = num_data - num_training
    print('Instances in test data: ', num_test)
    test_df = df[num_training:]
    print(test_df)
    
    return training_df, test_df

# Check data.
def check_data(df: pd.DataFrame) -> None:
    '''
    Display the summary statistics, data types and a plot of the data.

    Parameters:
        df (pd.DataFrame): The DataFrame to be checked and visualised.
    '''
    print(df, '\n')
    print(df.describe(), '\n')
    print(df.dtypes, '\n')
    plt.plot(df['date'], df['close'])

# Preprocessing.
def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    '''
    Preprocess the DataFrame.

    Parameters:
        df (pd.DataFrame): The DataFrame to be preprocessed.

    Returns:
        pd.DataFrame: The preprocessed DataFrame.
    '''
    # Convert dates to datetime format.
    df.loc[:, 'date'] = pd.to_datetime(df['date'], dayfirst = True)
    print(df['date'].dtypes)
    
    # Remove missing values.
    num_null = df.isnull().sum()
    if num_null.all() == 0:
        print('No data missing in dataframe.')
    elif num_null.any() > 0:
        df = df.dropna()
        print('Number of missing data removed: ', num_null)
    
    # Scale data.
    scaler = MinMaxScaler()
    numeric_cols = ['open', 'high', 'low', 'close', 'adj_close', 'volume']
    df.loc[:, numeric_cols] = scaler.fit_transform(df[numeric_cols])
    
    return df

# Sequence generation.
def generate_sequences(df: pd.DataFrame, sequence_length: int) -> tuple:
    '''
    Generate input sequences and target values.

    Parameters:
        df (pd.DataFrame): The DataFrame containing the data.
        sequence_length (int): The length of input sequences.

    Returns:
        tuple: A tuple containing two arrays: (input_sequences, target_values).
            - input_sequences (np.array): Array of input sequences.
            - target_values (np.array): Array of target values.
    '''
    input_sequences = []
    target_values = []
    df = df.drop(columns = ['date'])
    for i in range(len(df) - sequence_length):
        input_sequences.append(df.iloc[i:i + sequence_length].values)
        target_values.append(df.iloc[i + sequence_length]['close'])
        
    input_sequences = np.array(input_sequences)
    target_values = np.array(target_values)
        
    return input_sequences, target_values

# Build RNN.
def build_rnn(sequence_length: int, num_features: int, dropout_rate: float) -> Sequential:
    '''
    Build and compile the Recurrent Neural Network.

    Parameters:
        sequence_length (int): The length of input sequences.
        num_features (int): The number of features in the input data.
        dropout_rate (float): The dropout rate for regularisation.

    Returns:
        Sequential: The compiled model.
    '''
    num_neurons = 20
    model = Sequential([
        SimpleRNN(num_neurons, activation = 'relu', input_shape = (sequence_length, num_features)),
        Dropout(dropout_rate), Dense(1)
    ])
    model.compile(optimizer = 'adam', loss = 'mse')
    
    return model

# Fit model.
def fit_model(model: Sequential, training_input_sequences: np.array,
              training_target_values: np.array, batch_size: int,
              epochs: int, validation_split: float) -> dict:
    '''
    Fit the model and return the training history.

    Parameters:
        model (Sequential): The RNN model to be trained.
        training_input_sequences (np.array): Array of input sequences for training.
        training_target_values (np.array): Array of target values for training.
        batch_size (int): The batch size for training.
        epochs (int): The number of epochs for training.
        validation_split (float): The fraction of training data to use for validation.

    Returns:
        dict: A dictionary containing the training history.
    '''
    history = model.fit(training_input_sequences, training_target_values,
                        batch_size = batch_size, epochs = epochs,
                        validation_split = validation_split)
    plt.plot(history.history['loss'], label = 'Training loss')
    plt.plot(history.history['val_loss'], label = 'Validation loss')
    plt.title('Training and Validation Loss Over Epochs', fontweight = 'bold')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    
    return history

# Evaluate model.
def evaluate_model(input_sequences: np.array, target_values: np.array) -> float:
    '''
    Evaluate the model on input sequences and target values.

    Parameters:
        model (Sequential): The trained RNN model.
        input_sequences (np.array): Array of input sequences for evaluation.
        target_values (np.array): Array of target values for evaluation.

    Returns:
        float: The evaluation loss.
    '''
    loss = model.evaluate(input_sequences, target_values, verbose = 0)
    
    return loss

# Make and visualise predictions.
def predict_with_model(input_sequences: np.array, target_values: np.array,
                       data_used: str) -> np.array:
    '''
    Make and visualise the predictions the model predictions.

    Parameters:
        model (Sequential): The trained RNN model.
        input_sequences (np.array): Array of input sequences for prediction.
        target_values (np.array): Array of target values for comparison.

    Returns:
        np.array: Array of predicted values.
    '''
    predictions = model.predict(input_sequences)
    
    plt.figure(figsize = (10, 6))
    plt.plot(target_values, label = 'Actual data')
    plt.plot(predictions, label = 'Predicted values')
    plt.title(f'Actual vs. Predicted Values ({data_used} Data)', fontweight = 'bold')
    plt.xlabel('Time')
    plt.ylabel('Day Close Stock Price')
    plt.legend()
    plt.show()
    
    return predictions

# Call functions to run model.
data_path = 'C:/Users/Oscar/Documents/Projects/rnn_sequence_prediction/google_stock_prices.csv'
df = load_data(data_path)

split_ratio = 0.7
training_df, test_df = split_data(df, split_ratio)

check_data(training_df)
check_data(test_df)

training_df = preprocess_data(training_df)
test_df = preprocess_data(test_df)

sequence_length = 30
num_features = len(training_df.columns) - 1  # Exclude 'date' column.
training_input_sequences, training_target_values = generate_sequences(training_df, sequence_length)
test_input_sequences, test_target_values = generate_sequences(test_df, sequence_length)

dropout_rate = 0.1
model = build_rnn(sequence_length, num_features, dropout_rate)

epochs = 50
batch_size = 64
validation_split = 0.2
history = fit_model(model, training_input_sequences, training_target_values, batch_size, epochs, validation_split)

training_loss = evaluate_model(training_input_sequences, training_target_values)
print(f'Training loss: {training_loss:.4f}')
test_loss = evaluate_model(test_input_sequences, test_target_values)
print(f'Test loss: {test_loss:.4f}')

training_predictions = predict_with_model(training_input_sequences, training_target_values, 'Training')
test_predictions = predict_with_model(test_input_sequences, test_target_values, 'Test')
