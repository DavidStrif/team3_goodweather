# 3_Model: Neural Network Model Development and Evaluation

This folder contains the machine learning model development work for the GoodWeather project, focusing on neural network approaches for sales prediction.

## Overview

The model development process involves data preparation, neural network training with iterative improvements, and evaluation of prediction performance. The folder includes multiple versions of the neural network implementation, showing the evolution of the model.

## Files and Notebooks

### Notebooks

- **[model_definition_evaluation.ipynb](model_definition_evaluation.ipynb)**  
  Template notebook for model definition and evaluation. Covers model selection, feature engineering, hyperparameter tuning, implementation, evaluation metrics, and comparative analysis.

- **[neural_net_data_preparation_adrian.ipynb](neural_net_data_preparation_adrian.ipynb)**  
  Data preparation notebook for neural network training. Handles data loading, categorical feature encoding, feature scaling, and preparation of training/validation/prediction datasets. Outputs processed data as pickle files in the `pickle_data/` subdirectory.

- **[neural_net_estimation_1.0.ipynb](neural_net_estimation_1.0.ipynb)**  
  Initial neural network implementation (version 1.0). Builds and trains a basic neural network model using the prepared data from pickle files.

- **[neural_net_estimation_2.0.ipynb](neural_net_estimation_2.0.ipynb)**  
  Improved neural network implementation (version 2.0). Contains enhancements and refinements over the 1.0 version.

- **[neural_net_estimation_3.0.ipynb](neural_net_estimation_3.0.ipynb)**  
  Final neural network implementation (version 3.0). The most advanced version with optimized architecture, training procedures, and evaluation metrics.

### Data Files

#### Pickle Data (`pickle_data/` subdirectory)
- `training_features.pkl` - Feature data for model training
- `training_labels.pkl` - Target labels for training
- `validation_features.pkl` - Feature data for model validation
- `validation_labels.pkl` - Target labels for validation
- `prediction_features.pkl` - Feature data for generating predictions
- `prediction_ids.pkl` - IDs corresponding to prediction instances

#### Model Files
- **good_weather_model.h5** - Saved trained neural network model in HDF5 format (Keras/TensorFlow compatible)

#### Prediction Outputs
- **neural_net_predictions_2018-2019.csv** - Model predictions for the 2018-2019 period (version unspecified)
- **neural_net_predictions_2018-2019_3.0.csv** - Model predictions for the 2018-2019 period (version 3.0)

## Workflow

1. **Data Preparation**: Run `neural_net_data_preparation_adrian.ipynb` to process raw data and create pickle files
2. **Model Development**: Use the estimation notebooks (1.0 → 2.0 → 3.0) to iteratively improve the neural network
3. **Model Saving**: The trained model is saved as `good_weather_model.h5`
4. **Prediction Generation**: Use the trained model to generate sales predictions for the target period

## Dependencies

The notebooks require the following Python libraries:
- pandas
- numpy
- scikit-learn
- tensorflow/keras
- matplotlib/seaborn (for visualization)

## Usage

1. Start with data preparation: `neural_net_data_preparation_adrian.ipynb`
2. Run the neural network estimation notebooks in order (1.0, 2.0, 3.0) to see the model evolution
3. Use the saved model (`good_weather_model.h5`) for inference or further analysis
4. Review prediction files for model outputs

## Evaluation

The neural network models predict sales (Umsatz) based on weather and other features. Performance should be evaluated using appropriate regression metrics such as RMSE, MAE, and R² score.
