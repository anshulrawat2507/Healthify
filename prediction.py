import os
import sys
import threading
import time

# PySpark is optional at runtime.
# On Python 3.12+, some PySpark code paths can fail to import due to the removal of `distutils`.
# This app has a rule-based fallback, so we should not hard-fail on import.
try:
    from pyspark.sql import SparkSession
    from pyspark.ml.classification import LogisticRegressionModel, RandomForestClassificationModel
    from pyspark.ml.linalg import Vectors
    _PYSPARK_AVAILABLE = True
except Exception as _pyspark_import_error:  # pragma: no cover
    SparkSession = None
    LogisticRegressionModel = None
    RandomForestClassificationModel = None
    Vectors = None
    _PYSPARK_AVAILABLE = False
    _PYSPARK_IMPORT_ERROR = _pyspark_import_error

# Get the project root directory (current directory)
HEALTHIFY_DIR = os.path.abspath(os.path.dirname(__file__))
# print(f"Project root directory: {HEALTHIFY_DIR}")
MODELS_DIR = os.path.join(HEALTHIFY_DIR, "models")

# Create models directory if it doesn't exist
os.makedirs(MODELS_DIR, exist_ok=True)

def initialize_spark(timeout=10):
    """Initialize Spark session with timeout"""
    if not _PYSPARK_AVAILABLE:
        return None

    # Create events for signaling
    spark_ready = threading.Event()
    spark_failed = threading.Event()
    spark_result = [None]  # Use list to store result from thread
    
    def init_spark_thread():
        try:
            print("Initializing Spark session...")
            spark = SparkSession.builder \
                .appName("HEALTHIFY - Prediction") \
                .config("spark.driver.memory", "2g") \
                .config("spark.python.worker.timeout", "600") \
                .getOrCreate()
            
            # Test if spark is working correctly
            test_df = spark.createDataFrame([(1,)], ["test"])
            test_df.count()  # Force evaluation
            
            spark_result[0] = spark
            spark_ready.set()
        except Exception as e:
            print(f"Error initializing Spark: {e}")
            print("Attempting fallback to a simpler configuration...")
            try:
                spark = SparkSession.builder \
                    .appName("HEALTHIFY - Prediction Minimal") \
                    .config("spark.driver.memory", "1g") \
                    .getOrCreate()
                
                # Test if spark is working correctly
                test_df = spark.createDataFrame([(1,)], ["test"])
                test_df.count()  # Force evaluation
                
                spark_result[0] = spark
                spark_ready.set()
            except Exception as e2:
                print(f"Error initializing Spark with minimal config: {e2}")
                spark_failed.set()
    
    # Start the initialization in a separate thread
    spark_thread = threading.Thread(target=init_spark_thread)
    spark_thread.daemon = True
    spark_thread.start()
    
    # Wait for either success or timeout
    start_time = time.time()
    while time.time() - start_time < timeout:
        if spark_ready.is_set():
            print(f"Spark initialized successfully in {time.time() - start_time:.2f} seconds")
            return spark_result[0]
        if spark_failed.is_set():
            print("Spark initialization failed")
            return None
        time.sleep(0.1)
    
    # If we reach here, we've timed out
    # print(f"Spark initialization timed out after {timeout} seconds")
    # print("Using rule-based fallback prediction instead")
    return None

def create_feature_vector(spark, features):
    """Create a DataFrame with feature vector"""
    if not _PYSPARK_AVAILABLE:
        raise RuntimeError(f"PySpark is unavailable: {_PYSPARK_IMPORT_ERROR}")
    # Create DataFrame with feature vector
    data = [(Vectors.dense(features),)]
    return spark.createDataFrame(data, ["features"])

def fallback_diabetes_prediction(features):
    """Rule-based fallback prediction when Spark fails"""
    try:
        # Extract important features
        pregnancies = features[0]  # Number of pregnancies
        glucose = features[1]      # Glucose level
        blood_pressure = features[2]  # Blood pressure
        skin_thickness = features[3]  # Skin thickness
        insulin = features[4]      # Insulin level
        bmi = features[5]          # BMI
        dpf = features[6]          # Diabetes pedigree function
        age = features[7]          # Age
        
        # Check if these are default values from the UI
        is_default_input = (pregnancies == 0 and glucose == 120 and blood_pressure == 70 and 
                         skin_thickness == 20 and insulin == 80 and 
                         bmi == 25.0 and dpf == 0.5 and age == 30)
        
        if is_default_input:
            # print("Default values detected, providing conservative estimate")  # Comment out this line
            # For default values, give a conservative estimate
            return 0, 0.20
    except Exception as e:
        # print(f"Error extracting features: {e}")  # Comment out this line
        # If any error in feature extraction, give conservative estimate
        return 0, 0.20
    
    # Start with a base probability
    probability = 0.1
    
    # Glucose has the strongest correlation with diabetes
    if glucose <= 90:
        probability += 0.05
    elif glucose <= 110:
        probability += 0.15
    elif glucose <= 125:
        probability += 0.3
    elif glucose <= 140:
        probability += 0.45
    elif glucose <= 160:
        probability += 0.55
    elif glucose <= 180:
        probability += 0.65
    else:
        probability += 0.75
    
    # BMI impact
    if bmi < 18.5:
        probability += 0.05  # Underweight - slight increase
    elif bmi < 25:
        probability += 0.0   # Normal weight - no change
    elif bmi < 30:
        probability += 0.1   # Overweight - moderate increase
    elif bmi < 35:
        probability += 0.15  # Obesity class I - significant increase
    elif bmi < 40:
        probability += 0.2   # Obesity class II - high increase
    else:
        probability += 0.25  # Obesity class III - very high increase
    
    # Age risk adjustment
    if age < 35:
        probability -= 0.05  # Lower risk for young age
    elif age < 45:
        probability += 0.0   # Neutral for middle age
    elif age < 55:
        probability += 0.1   # Increased risk for age 45-54
    elif age < 65:
        probability += 0.15  # Higher risk for age 55-64
    else:
        probability += 0.2   # Highest risk for age 65+
    
    # Diabetes Pedigree Function impact (family history)
    if dpf > 0.8:
        probability += 0.15  # Strong family history
    elif dpf > 0.5:
        probability += 0.07  # Moderate family history
    
    # Blood pressure impact
    if blood_pressure > 140:
        probability += 0.1   # High blood pressure increases risk
    
    # Insulin levels impact
    if insulin < 15 and glucose > 125:
        probability += 0.2   # Low insulin with high glucose suggests insulin resistance
    
    # Ensure probability is between 0.05 and 0.95
    probability = max(0.05, min(0.95, probability))
    
    # Determine result based on final probability
    result = 1 if probability >= 0.5 else 0
    
    return result, probability

def fallback_heart_prediction(features):
    """Rule-based fallback prediction for heart attack risk"""
    try:
        # Extract important features
        age = features[0] if len(features) > 0 else 50
        gender = features[1] if len(features) > 1 else 0     # 1 for male, 0 for female
        diabetes = features[2] if len(features) > 2 else 0   # 1 for yes, 0 for no
        hypertension = features[3] if len(features) > 3 else 0  # 1 for yes, 0 for no
        obesity = features[4] if len(features) > 4 else 0    # 1 for yes, 0 for no
        smoking = features[5] if len(features) > 5 else 0    # 1 for yes, 0 for no
        alcohol = features[6] if len(features) > 6 else 0    # 1 for yes, 0 for no
        physical_activity = features[7] if len(features) > 7 else 0  # 1 for yes, 0 for no
        diet_score = features[8] if len(features) > 8 else 5  # Diet score
        
        # Check if these are default values from the UI
        is_default_input = (age == 40 and gender == 0 and 
                         diabetes == 0 and hypertension == 0 and
                         obesity == 0 and smoking == 0 and
                         alcohol == 0 and physical_activity == 0 and
                         diet_score == 5)
        
        if is_default_input:
            print("Default values detected, providing conservative estimate")
            # For default values, give a conservative estimate
            return 0, 0.20
        
        # Get cholesterol related values if available
        cholesterol = features[9] if len(features) > 9 else 200
        hdl = features[12] if len(features) > 12 else 50
        ldl = features[11] if len(features) > 11 else 100
        
        # Get blood pressure if available
        systolic_bp = features[13] if len(features) > 13 else 120
        diastolic_bp = features[14] if len(features) > 14 else 80
        
        # Get other risk factors
        family_history = features[16] if len(features) > 16 else 0  # 1 for yes, 0 for no
        stress = features[17] if len(features) > 17 else 5  # 1-10 scale
    except Exception as e:
        print(f"Error extracting features: {e}")
        # Default values if extraction fails
        return 0, 0.20
    
    # Start with a base probability
    probability = 0.15
    
    # Age impact (one of the strongest predictors)
    if age < 40:
        probability -= 0.05
    elif age < 50:
        probability += 0.05
    elif age < 60:
        probability += 0.15
    elif age < 70:
        probability += 0.25
    else:
        probability += 0.3
    
    # Gender impact (males have higher risk)
    if gender == 1:  # Male
        probability += 0.1
    
    # Diabetes impact
    if diabetes == 1:
        probability += 0.15
    
    # Hypertension impact
    if hypertension == 1:
        probability += 0.15
    
    # Obesity impact
    if obesity == 1:
        probability += 0.1
    
    # Smoking impact (major risk factor)
    if smoking == 1:
        probability += 0.2
    
    # Cholesterol impact
    if cholesterol > 240:
        probability += 0.15
    elif cholesterol > 200:
        probability += 0.1
    
    # Physical activity impact (protective)
    if physical_activity == 1:
        probability -= 0.1
    
    # Stress impact
    if stress >= 8:
        probability += 0.15
    elif stress >= 5:
        probability += 0.05
    
    # Family history impact
    if family_history == 1:
        probability += 0.15
    
    # Ensure probability is between 0.05 and 0.95
    probability = max(0.05, min(0.95, probability))
    
    # Determine result based on final probability
    result = 1 if probability >= 0.5 else 0
    
    return result, probability

def predict_diabetes(features):
    """Predict diabetes risk using trained ML models with fallback"""
    # Get rule-based prediction for fallback
    rule_prediction, rule_probability = fallback_diabetes_prediction(features)
    
    # Create a simplified result structure
    final_result = {
        "prediction": rule_prediction,
        "probability": rule_probability,
        "model_used": "rule_based",
        "details": {
            "logistic_regression": {"prediction": None, "probability": None},
            "random_forest": {"prediction": None, "probability": None},
            "rule_based": {"prediction": rule_prediction, "probability": rule_probability}
        }
    }
    
    # Try using Spark models with timeout
    try:
        # Initialize Spark with 10 second timeout
        spark = initialize_spark(timeout=10)
        if spark is None:
            # Silently fail - we already have rule-based results
            return final_result
            
        # Create feature vector
        df = create_feature_vector(spark, features)
        
        # Try to use the best model we have
        best_model_used = False
        
        # Check if we have a best diabetes model
        best_models_path = os.path.join(MODELS_DIR, "best_models.txt")
        if os.path.exists(best_models_path):
            with open(best_models_path, "r") as f:
                for line in f:
                    if line.startswith("Best diabetes model:"):
                        best_model_path = line.split(":", 1)[1].strip()
                        
                        # Check which model type it is
                        if "lr" in best_model_path.lower():
                            model_type = "logistic_regression"
                        elif "rf" in best_model_path.lower():
                            model_type = "random_forest"
                        else:
                            model_type = "unknown"
                            
                        # Try to load and use the best model
                        if os.path.exists(best_model_path):
                            try:
                                if "lr" in best_model_path.lower():
                                    model = LogisticRegressionModel.load(best_model_path)
                                else:
                                    model = RandomForestClassificationModel.load(best_model_path)
                                
                                predictions = model.transform(df)
                                pred_row = predictions.select("prediction", "probability").collect()[0]
                                prediction = int(pred_row["prediction"])
                                probability = float(pred_row["probability"][1])
                                
                                # Update result with best model prediction
                                final_result["prediction"] = prediction
                                final_result["probability"] = probability
                                final_result["model_used"] = model_type
                                final_result["details"][model_type] = {
                                    "prediction": prediction,
                                    "probability": probability
                                }
                                
                                best_model_used = True
                                break
                            except Exception as e:
                                # Silently fail, we'll try individual models
                                pass
        
        # If best model didn't work, try individual models
        if not best_model_used:
            # Try logistic regression
            lr_model_path = os.path.join(MODELS_DIR, "diabetes_model_lr")
            if os.path.exists(lr_model_path):
                try:
                    lr_model = LogisticRegressionModel.load(lr_model_path)
                    lr_predictions = lr_model.transform(df)
                    
                    lr_pred_row = lr_predictions.select("prediction", "probability").collect()[0]
                    lr_prediction = int(lr_pred_row["prediction"])
                    lr_probability = float(lr_pred_row["probability"][1])
                    
                    # Update result with LR model
                    final_result["prediction"] = lr_prediction
                    final_result["probability"] = lr_probability
                    final_result["model_used"] = "logistic_regression"
                    final_result["details"]["logistic_regression"] = {
                        "prediction": lr_prediction,
                        "probability": lr_probability
                    }
                except Exception as e:
                    # Silently fail, we'll try random forest
                    pass
            
            # Try random forest if logistic regression didn't update the result
            if final_result["model_used"] == "rule_based":
                rf_model_path = os.path.join(MODELS_DIR, "diabetes_model_rf")
                if os.path.exists(rf_model_path):
                    try:
                        rf_model = RandomForestClassificationModel.load(rf_model_path)
                        rf_predictions = rf_model.transform(df)
                        
                        rf_pred_row = rf_predictions.select("prediction", "probability").collect()[0]
                        rf_prediction = int(rf_pred_row["prediction"])
                        rf_probability = float(rf_pred_row["probability"][1])
                        
                        # Update result with RF model
                        final_result["prediction"] = rf_prediction
                        final_result["probability"] = rf_probability
                        final_result["model_used"] = "random_forest"
                        final_result["details"]["random_forest"] = {
                            "prediction": rf_prediction,
                            "probability": rf_probability
                        }
                    except Exception as e:
                        # Silently fail, we have fallback predictions
                        pass
            
        # Stop Spark session
        spark.stop()
        
    except Exception as e:
        # Silently fail and use rule-based prediction
        pass
    
    return final_result

def predict_heart_attack(features):
    """Predict heart attack risk using trained ML models with fallback"""
    # Get rule-based prediction for fallback
    rule_prediction, rule_probability = fallback_heart_prediction(features)
    
    # Create a simplified result structure
    final_result = {
        "prediction": rule_prediction,
        "probability": rule_probability,
        "model_used": "rule_based",
        "details": {
            "logistic_regression": {"prediction": None, "probability": None},
            "random_forest": {"prediction": None, "probability": None},
            "rule_based": {"prediction": rule_prediction, "probability": rule_probability}
        }
    }
    
    # Try using Spark models with timeout
    try:
        # Initialize Spark with 10 second timeout
        spark = initialize_spark(timeout=10)
        if spark is None:
            # Silently fail - we already have rule-based results
            return final_result
            
        # Create feature vector
        df = create_feature_vector(spark, features)
        
        # Try to use the best model we have
        best_model_used = False
        
        # Check if we have a best heart model
        best_models_path = os.path.join(MODELS_DIR, "best_models.txt")
        if os.path.exists(best_models_path):
            with open(best_models_path, "r") as f:
                for line in f:
                    if line.startswith("Best heart model:"):
                        best_model_path = line.split(":", 1)[1].strip()
                        
                        # Check which model type it is
                        if "lr" in best_model_path.lower():
                            model_type = "logistic_regression"
                        elif "rf" in best_model_path.lower():
                            model_type = "random_forest"
                        else:
                            model_type = "unknown"
                            
                        # Try to load and use the best model
                        if os.path.exists(best_model_path):
                            try:
                                if "lr" in best_model_path.lower():
                                    model = LogisticRegressionModel.load(best_model_path)
                                else:
                                    model = RandomForestClassificationModel.load(best_model_path)
                                
                                predictions = model.transform(df)
                                pred_row = predictions.select("prediction", "probability").collect()[0]
                                prediction = int(pred_row["prediction"])
                                probability = float(pred_row["probability"][1])
                                
                                # Update result with best model prediction
                                final_result["prediction"] = prediction
                                final_result["probability"] = probability
                                final_result["model_used"] = model_type
                                final_result["details"][model_type] = {
                                    "prediction": prediction,
                                    "probability": probability
                                }
                                
                                best_model_used = True
                                break
                            except Exception as e:
                                # Silently fail, we'll try individual models
                                pass
        
        # If best model didn't work, try individual models
        if not best_model_used:
            # Try logistic regression
            lr_model_path = os.path.join(MODELS_DIR, "heart_model_lr")
            if os.path.exists(lr_model_path):
                try:
                    lr_model = LogisticRegressionModel.load(lr_model_path)
                    lr_predictions = lr_model.transform(df)
                    
                    lr_pred_row = lr_predictions.select("prediction", "probability").collect()[0]
                    lr_prediction = int(lr_pred_row["prediction"])
                    lr_probability = float(lr_pred_row["probability"][1])
                    
                    # Update result with LR model
                    final_result["prediction"] = lr_prediction
                    final_result["probability"] = lr_probability
                    final_result["model_used"] = "logistic_regression"
                    final_result["details"]["logistic_regression"] = {
                        "prediction": lr_prediction,
                        "probability": lr_probability
                    }
                except Exception as e:
                    # Silently fail, we'll try random forest
                    pass
            
            # Try random forest if logistic regression didn't update the result
            if final_result["model_used"] == "rule_based":
                rf_model_path = os.path.join(MODELS_DIR, "heart_model_rf")
                if os.path.exists(rf_model_path):
                    try:
                        rf_model = RandomForestClassificationModel.load(rf_model_path)
                        rf_predictions = rf_model.transform(df)
                        
                        rf_pred_row = rf_predictions.select("prediction", "probability").collect()[0]
                        rf_prediction = int(rf_pred_row["prediction"])
                        rf_probability = float(rf_pred_row["probability"][1])
                        
                        # Update result with RF model
                        final_result["prediction"] = rf_prediction
                        final_result["probability"] = rf_probability
                        final_result["model_used"] = "random_forest"
                        final_result["details"]["random_forest"] = {
                            "prediction": rf_prediction,
                            "probability": rf_probability
                        }
                    except Exception as e:
                        # Silently fail, we have fallback predictions
                        pass
            
        # Stop Spark session
        spark.stop()
        
    except Exception as e:
        # Silently fail and use rule-based prediction
        pass
    
    return final_result

# Test function for direct execution of this file
if __name__ == "__main__":
    # Sample diabetes prediction
    diabetes_features = [1, 140, 80, 30, 45, 27.5, 0.5, 45]
    diabetes_results = predict_diabetes(diabetes_features)
    print("Diabetes Prediction Results:", diabetes_results)
    
    # Sample heart attack prediction
    heart_features = [50, 1, 0, 1, 0, 1, 0, 1, 6, 220, 180, 120, 45, 145, 95]
    heart_results = predict_heart_attack(heart_features)
    print("Heart Attack Prediction Results:", heart_results) 