# 🩺 HEALTHIFY - Smart Disease Prediction System

## 🏥 Overview

HEALTHIFY is a smart, ML-based web application that helps users predict the risk of diabetes and heart attack. Built with Streamlit and powered by PySpark, it delivers fast and accurate predictions using pre-trained Logistic Regression and Random Forest models.

## ✨ Features

- **Diabetes Risk Prediction**: Predicts diabetes risk based on medical parameters
- **Heart Attack Risk Prediction**: Assesses heart attack risk using health data
- **Pre-trained Models**: Uses trained Logistic Regression and Random Forest models
- **Real-time Predictions**: Instant risk assessment with probability scores
- **User-friendly Interface**: Clean web interface built with Streamlit

## 🛠️ Technologies Used

- **Apache Spark**: Distributed computing framework
- **PySpark MLlib**: Machine learning library
- **Streamlit**: Web application framework
- **Python**: Programming language

## 📁 Project Structure
```
Healthify/
├── app.py # Main Streamlit application
├── prediction.py # Prediction logic and model loading
├── data_processing.py # Data preprocessing
├── model_training.py # Model training (optional)
├── run.py # Application runner
├── requirements.txt # Python dependencies
├── data/ # Dataset directory
│ ├── diabetes.csv
│ └── heart_attack_prediction_india.csv
└── models/ # Pre-trained models (included)
```

## 🚀 Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/Vegeta909/Healthify.git
   cd Healthify
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## 🎯 Usage

### Quick Start
```bash
streamlit run app.py
```

### Alternative
```bash
python run.py
```

The application will automatically load the pre-trained models and start the web interface.

## 📊 Datasets

- **Diabetes Dataset**: Pima Indians Diabetes Database
- **Heart Attack Dataset**: Heart Attack Prediction Dataset (India)

## 🤖 Pre-trained Models

The application includes pre-trained models for both diabetes and heart attack prediction:
- Logistic Regression models
- Random Forest models

**Note**: No model training is required as trained models are already included in the `models/` directory.

## 📈 Input Features

### Diabetes Prediction
- Pregnancies, Glucose, Blood Pressure, Skin Thickness
- Insulin, BMI, Diabetes Pedigree Function, Age

### Heart Attack Prediction
- Age, Gender, Medical conditions, Lifestyle factors
- Diet score, Family history, Stress levels
