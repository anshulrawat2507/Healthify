# 🩺 HEALTHIFY - Smart Disease Prediction System

## 🏥 Overview

HEALTHIFY is an intelligent Big Data-driven disease prediction system that leverages Apache Spark and machine learning to assess diabetes and heart attack risks. The application employs multiple model architectures including Logistic Regression and Random Forest with comprehensive fallback mechanisms, delivering accurate predictions through an intuitive Streamlit web interface.

**🚨 Medical Disclaimer**: This tool is for educational and screening purposes only and should not replace professional medical consultation.

## ✨ Key Features

### 🩺 **Dual Disease Prediction Engine**
- **Diabetes Risk Assessment**: Analyzes 8 medical parameters with 78.4% accuracy
- **Heart Attack Risk Evaluation**: Considers 14+ health factors with 82.3% accuracy
- **Probability Scoring**: Provides detailed risk percentages and confidence levels

### 🤖 **Advanced ML Architecture**
- **Multiple Algorithms**: Logistic Regression and Random Forest with automated model selection
- **Cross-Validation**: 3-fold validation with hyperparameter tuning for optimal performance
- **Fallback System**: Rule-based prediction when Spark models are unavailable
- **Model Comparison**: Automatic selection based on AUC performance metrics

### 🚀 **System Reliability**
- **Graceful Degradation**: Conservative estimation system maintains functionality
- **Error Handling**: Comprehensive error management with user experience continuity
- **Performance Optimization**: 2GB driver memory allocation with garbage collection monitoring

### 🖥️ **User Experience**
- **Real-time Predictions**: Instant risk assessment with sub-10 second response times
- **Interactive Interface**: Gender-specific form logic and input validation
- **Risk Visualization**: Color-coded risk levels with medical recommendations

## 🛠️ Technology Stack

| Component | Technology | Version | Purpose |
|-----------|------------|---------|---------|
| **Machine Learning** | PySpark MLlib | 3.5.3 | Distributed model training and evaluation |
| **Web Framework** | Streamlit | 1.32.0 | Interactive user interface |
| **Data Processing** | Apache Spark | 3.5.3 | Big data handling and preprocessing |
| **Core Language** | Python | 3.8+ | Application development |
| **Data Analysis** | Pandas | 2.1.4 | Data manipulation and analysis |
| **Numerical Computing** | NumPy | 1.26.3 | Mathematical operations |

## 📊 Model Performance Metrics

### Current Model Performance
| Model | Algorithm | Accuracy | Precision | Recall | F1-Score | AUC-ROC |
|-------|-----------|----------|-----------|--------|----------|---------|
| **Diabetes Prediction** | Logistic Regression | 78.4% | 72.1% | 69.8% | 0.709 | 0.812 |
| **Heart Attack Prediction** | Logistic Regression | 82.3% | 80.1% | 75.6% | 0.778 | 0.845 |
| **Fallback System** | Rule-based | 65.4% | 61.2% | 58.9% | 0.600 | 0.687 |

### Evaluation Criteria
- **Primary Metric**: AUC-ROC (Area Under Curve) for binary classification performance
- **Minimum Threshold**: 70% AUC for acceptable performance
- **Target Performance**: 80%+ AUC for production deployment
- **Clinical Relevance**: Interpretable probability scores with medical recommendations

## 📁 Project Structure

```
Healthify/
├── app.py                    # Streamlit web application (12KB, 299 lines)
├── prediction.py             # Prediction logic and fallback system (24KB, 561 lines)
├── run.py                    # Application runner (11KB, 266 lines)
├── model_training.py         # Model training and evaluation (12KB, 288 lines)
├── data_processing.py        # Data preprocessing pipeline (13KB, 341 lines)
├── requirements.txt          # Python dependencies (105B, 7 lines)
├── models/                   # Trained model storage
│   ├── diabetes_model_lr/    # Best diabetes model
│   ├── diabetes_model_rf/    # Alternative diabetes model
│   ├── heart_model_lr/       # Best heart attack model
│   ├── heart_model_rf/       # Alternative heart attack model
│   └── best_models.txt       # Model selection record
└── data/                     # Dataset storage
    ├── diabetes.csv          # Raw diabetes dataset
    ├── heart_attack_prediction_india.csv  # Raw heart attack dataset
    └── processed/            # Preprocessed data in Parquet format
```

## 🚀 Installation & Setup

### Prerequisites
- **Python 3.8+** with pip package manager
- **Java 8+** (required for PySpark operations)
- **Git** for repository cloning

### Step-by-Step Installation

1. **Clone Repository**
   ```bash
   git clone https://github.com/Vegeta909/Healthify.git
   cd Healthify
   ```

2. **Create Virtual Environment** (Recommended)
   ```bash
   # Create virtual environment
   python -m venv healthify_env
   
   # Activate environment
   # Windows:
   healthify_env\Scripts\activate
   # macOS/Linux:
   source healthify_env/bin/activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify Installation**
   ```bash
   python -c "import pyspark; print('PySpark installed successfully')"
   ```

## 🎯 Usage Guide

### Quick Start
```bash
# Launch the web application
streamlit run app.py
```

### Alternative Launch Method
```bash
# Using the application runner
python run.py
```

### Accessing the Application
- **Local URL**: `http://localhost:8501`
- **Network URL**: Available for local network access

### Using the Prediction Interface

#### For Diabetes Risk Assessment:
1. Navigate to the "Diabetes Prediction" tab
2. Enter health parameters:
   - Pregnancies (0-20), Glucose level (mg/dL), Blood pressure (mm Hg)
   - BMI, Age, Insulin levels, and other medical indicators
3. Click "Predict Diabetes Risk" for instant assessment
4. Review risk probability and medical recommendations

#### For Heart Attack Risk Assessment:
1. Switch to "Heart Attack Prediction" tab
2. Complete comprehensive health profile:
   - Demographics, medical history, lifestyle factors
   - Diet score, family history, stress levels
3. Submit for immediate risk evaluation
4. Receive personalized health guidance

## 📈 Detailed Input Features

### Diabetes Prediction Parameters
| Feature | Description | Range | Medical Significance |
|---------|-------------|-------|---------------------|
| **Pregnancies** | Number of pregnancies | 0-20 | Gestational diabetes risk factor |
| **Glucose** | Plasma glucose concentration | 0-300 mg/dL | Primary diabetes indicator |
| **Blood Pressure** | Diastolic blood pressure | 0-200 mm Hg | Cardiovascular health marker |
| **Skin Thickness** | Triceps skin fold thickness | 0-100 mm | Body fat distribution indicator |
| **Insulin** | 2-hour serum insulin | 0-900 mu U/ml | Insulin resistance marker |
| **BMI** | Body Mass Index | 0.0-70.0 | Obesity assessment |
| **Diabetes Pedigree** | Family history function | 0.0-3.0 | Genetic predisposition |
| **Age** | Age in years | 0-120 | Age-related risk factor |

### Heart Attack Risk Parameters
| Category | Features | Clinical Importance |
|----------|----------|-------------------|
| **Demographics** | Age, Gender | Basic risk stratification |
| **Medical History** | Diabetes, Hypertension, Obesity | Established cardiovascular risk factors |
| **Lifestyle** | Smoking, Alcohol, Physical Activity | Modifiable risk factors |
| **Laboratory Values** | Cholesterol, Triglycerides, HDL/LDL | Lipid profile assessment |
| **Clinical Measurements** | Systolic/Diastolic BP | Blood pressure monitoring |
| **Environmental** | Air Pollution Exposure | Environmental risk factors |
| **Psychosocial** | Stress Level, Healthcare Access | Comprehensive risk assessment |

## 🔬 Model Training & Evaluation

### Model Selection Process
- **Automated Comparison**: Both Logistic Regression and Random Forest models are trained
- **AUC-Based Selection**: Best model selected based on Area Under Curve performance
- **Current Winners**: Logistic Regression models selected for both prediction tasks

### Data Processing Features
- **Zero-Value Imputation**: Medical impossibilities replaced with column means
- **Feature Standardization**: StandardScaler ensures equal feature contribution
- **Categorical Encoding**: Gender and other categorical variables properly encoded
- **Parquet Storage**: Optimized I/O performance with compressed storage

## 🛡️ System Reliability Features

### Fallback Mechanisms
- **Rule-Based Prediction**: Available when Spark models fail
- **Conservative Estimation**: Default probability of 20% for unknown cases
- **Graceful Degradation**: System maintains functionality under constraints

### Error Handling
- **Input Validation**: Comprehensive range checking and data type validation
- **Gender-Specific Logic**: Pregnancy input disabled for male users
- **Memory Management**: Optimized Spark configuration with garbage collection

## 📋 Dependencies

```txt
pyspark==3.5.3
streamlit==1.32.0
pandas==2.1.4
numpy==1.26.3
matplotlib==3.8.2
scikit-learn==1.4.0
```

---

**Made for accessible healthcare prediction and early disease detection**
