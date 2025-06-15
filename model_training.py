from pyspark.sql import SparkSession
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml.linalg import VectorUDT, Vectors
import time
import os
import sys

# Get the project root directory (current directory)
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
print(f"Project root directory: {PROJECT_ROOT}")

# Create necessary directories if they don't exist
os.makedirs(os.path.join(PROJECT_ROOT, "models"), exist_ok=True)
os.makedirs(os.path.join(PROJECT_ROOT, "data", "processed"), exist_ok=True)

# Only set Hadoop/Java paths if they exist
hadoop_path = r'C:\hadoop'
if os.path.exists(hadoop_path):
    # Set HADOOP_HOME environment variable
    os.environ['HADOOP_HOME'] = hadoop_path
    print(f"Found Hadoop at: {hadoop_path}")
    
    # Add Hadoop bin to PATH
    hadoop_bin = os.path.join(os.environ['HADOOP_HOME'], 'bin')
    if os.path.exists(hadoop_bin):
        os.environ['PATH'] = hadoop_bin + os.pathsep + os.environ['PATH']
        
        # Set hadoop.home.dir system property for Java
        if 'JAVA_OPTS' not in os.environ:
            os.environ['JAVA_OPTS'] = ''
        os.environ['JAVA_OPTS'] = os.environ['JAVA_OPTS'] + ' -Dhadoop.home.dir=C:\\hadoop'
    else:
        print(f"Warning: Hadoop bin directory not found at {hadoop_bin}")
else:
    print(f"Warning: Hadoop directory not found at {hadoop_path}")
    print("Continuing without Hadoop configuration...")

def initialize_spark():
    """Initialize Spark session"""
    try:
        print("Initializing Spark session...")
        # Add more configurations to help debug issues
        spark = SparkSession.builder \
            .appName("HEALTHIFY - Model Training") \
            .config("spark.driver.memory", "2g") \
            .config("spark.driver.extraJavaOptions", "-XX:+PrintGCDetails -XX:+PrintGCTimeStamps") \
            .config("spark.python.worker.reuse", "false") \
            .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
            .getOrCreate()
            
        return spark
    except Exception as e:
        print(f"Error initializing Spark: {e}")
        print("Attempting to fallback to a simpler Spark configuration...")
        
        try:
            # Try with minimal config
            spark = SparkSession.builder \
                .appName("HEALTHIFY - Minimal") \
                .config("spark.driver.memory", "1g") \
                .getOrCreate()
            return spark
        except Exception as e2:
            print(f"Fatal error initializing Spark with minimal config: {e2}")
            print("Cannot proceed without Spark. Please check your Java/Spark installation.")
            sys.exit(1)

def load_processed_data(spark, name):
    """Load processed data from parquet files"""
    # Load the data
    data_path = os.path.join(PROJECT_ROOT, "data", "processed", name)
    print(f"Loading processed data from {data_path}")
    
    if not os.path.exists(data_path):
        print(f"Error: Processed data not found at {data_path}")
        fallback_path = os.path.join(PROJECT_ROOT, "data", "processed", f"{name}_pandas.csv")
        if os.path.exists(fallback_path):
            print(f"Found pandas-processed data at {fallback_path}, trying to load it instead")
            return spark.read.option("header", "true").option("inferSchema", "true").csv(fallback_path)
        else:
            print(f"No fallback data found at {fallback_path}")
            sys.exit(1)
    
    df = spark.read.parquet(data_path)
    
    # If the features column is not a proper Vector type, convert it
    if "features" in df.columns:
        from pyspark.sql.types import StructType
        features_type = df.schema["features"].dataType
        
        if isinstance(features_type, StructType):
            # Convert the struct back to a Vector
            from pyspark.sql.functions import udf
            
            def to_vector(struct):
                if struct is None:
                    return None
                return Vectors.dense(struct["values"])
            
            to_vector_udf = udf(to_vector, VectorUDT())
            df = df.withColumn("features", to_vector_udf("features"))
    
    return df

def train_logistic_regression(train_data, test_data, label_col):
    """Train logistic regression model with hyperparameter tuning"""
    start_time = time.time()
    print(f"Training logistic regression model for {label_col}...")
    
    # Create logistic regression model
    lr = LogisticRegression(featuresCol="features", labelCol=label_col, maxIter=10)
    
    # Create parameter grid for hyperparameter tuning
    paramGrid = ParamGridBuilder() \
        .addGrid(lr.regParam, [0.01, 0.1, 0.3]) \
        .addGrid(lr.elasticNetParam, [0.0, 0.5, 1.0]) \
        .build()
    
    # Create binary evaluator
    evaluator = BinaryClassificationEvaluator(labelCol=label_col)
    
    # Create cross validator
    cv = CrossValidator(estimator=lr,
                       estimatorParamMaps=paramGrid,
                       evaluator=evaluator,
                       numFolds=3)
    
    # Train model with cross validation
    cv_model = cv.fit(train_data)
    best_model = cv_model.bestModel
    
    # Get training time
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")
    
    # Evaluate model on test data
    predictions = best_model.transform(test_data)
    auc = evaluator.evaluate(predictions)
    print(f"AUC on test data: {auc:.4f}")
    
    # Print best parameters
    print(f"Best regParam: {best_model.getRegParam()}")
    print(f"Best elasticNetParam: {best_model.getElasticNetParam()}")
    
    return best_model

def train_random_forest(train_data, test_data, label_col):
    """Train random forest model with hyperparameter tuning"""
    start_time = time.time()
    print(f"Training random forest model for {label_col}...")
    
    # Create random forest model
    rf = RandomForestClassifier(featuresCol="features", labelCol=label_col, seed=42)
    
    # Create parameter grid for hyperparameter tuning
    paramGrid = ParamGridBuilder() \
        .addGrid(rf.numTrees, [10, 20]) \
        .addGrid(rf.maxDepth, [5, 10]) \
        .build()
    
    # Create binary evaluator
    evaluator = BinaryClassificationEvaluator(labelCol=label_col)
    
    # Create cross validator
    cv = CrossValidator(estimator=rf,
                       estimatorParamMaps=paramGrid,
                       evaluator=evaluator,
                       numFolds=3)
    
    # Train model with cross validation
    cv_model = cv.fit(train_data)
    best_model = cv_model.bestModel
    
    # Get training time
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")
    
    # Evaluate model on test data
    predictions = best_model.transform(test_data)
    auc = evaluator.evaluate(predictions)
    print(f"AUC on test data: {auc:.4f}")
    
    # Print best parameters and feature importances
    print(f"Best numTrees: {best_model.getNumTrees}")
    print(f"Best maxDepth: {best_model.getMaxDepth()}")
    print(f"Feature importances: {best_model.featureImportances}")
    
    # Calculate accuracy using MulticlassClassificationEvaluator
    multi_evaluator = MulticlassClassificationEvaluator(labelCol=label_col, metricName="accuracy")
    accuracy = multi_evaluator.evaluate(predictions)
    print(f"Accuracy on test data: {accuracy:.4f}")
    
    return best_model

def save_model(model, model_path):
    """Save model to disk"""
    # Ensure the directory exists
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    model.write().overwrite().save(model_path)
    print(f"Model saved to {model_path}")

def main():
    """Main function to train models"""
    # Initialize Spark
    spark = initialize_spark()
    
    # Create directories if they don't exist
    models_dir = os.path.join(PROJECT_ROOT, "models")
    os.makedirs(models_dir, exist_ok=True)
    
    # Train diabetes model
    print("\n=== Training Diabetes Model ===")
    diabetes_data = load_processed_data(spark, "diabetes")
    diabetes_train_data, diabetes_test_data = diabetes_data.randomSplit([0.8, 0.2], seed=42)
    
    # Train logistic regression model for diabetes
    diabetes_lr_model = train_logistic_regression(diabetes_train_data, diabetes_test_data, "Outcome")
    save_model(diabetes_lr_model, os.path.join(models_dir, "diabetes_model_lr"))
    
    # Train random forest model for diabetes
    diabetes_rf_model = train_random_forest(diabetes_train_data, diabetes_test_data, "Outcome")
    save_model(diabetes_rf_model, os.path.join(models_dir, "diabetes_model_rf"))
    
    # Train heart attack model
    print("\n=== Training Heart Attack Model ===")
    heart_data = load_processed_data(spark, "heart")
    heart_train_data, heart_test_data = heart_data.randomSplit([0.8, 0.2], seed=42)
    
    # Train logistic regression model for heart attack
    heart_lr_model = train_logistic_regression(heart_train_data, heart_test_data, "Heart_Attack_Risk")
    save_model(heart_lr_model, os.path.join(models_dir, "heart_model_lr"))
    
    # Train random forest model for heart attack
    heart_rf_model = train_random_forest(heart_train_data, heart_test_data, "Heart_Attack_Risk")
    save_model(heart_rf_model, os.path.join(models_dir, "heart_model_rf"))
    
    # Compare models and select the best one for each prediction task
    print("\n=== Model Comparison ===")
    
    # For diabetes prediction - use diabetes_test_data
    diabetes_lr_eval = BinaryClassificationEvaluator(labelCol="Outcome").evaluate(
        diabetes_lr_model.transform(diabetes_test_data))
    diabetes_rf_eval = BinaryClassificationEvaluator(labelCol="Outcome").evaluate(
        diabetes_rf_model.transform(diabetes_test_data))
    
    print(f"Diabetes - Logistic Regression AUC: {diabetes_lr_eval:.4f}")
    print(f"Diabetes - Random Forest AUC: {diabetes_rf_eval:.4f}")
    
    if diabetes_rf_eval > diabetes_lr_eval:
        print("Random Forest performs better for diabetes prediction")
        best_diabetes_model = diabetes_rf_model
        best_diabetes_model_path = os.path.join(models_dir, "diabetes_model_rf")
    else:
        print("Logistic Regression performs better for diabetes prediction")
        best_diabetes_model = diabetes_lr_model
        best_diabetes_model_path = os.path.join(models_dir, "diabetes_model_lr")
    
    # For heart attack prediction - use heart_test_data
    heart_lr_eval = BinaryClassificationEvaluator(labelCol="Heart_Attack_Risk").evaluate(
        heart_lr_model.transform(heart_test_data))
    heart_rf_eval = BinaryClassificationEvaluator(labelCol="Heart_Attack_Risk").evaluate(
        heart_rf_model.transform(heart_test_data))
    
    print(f"Heart Attack - Logistic Regression AUC: {heart_lr_eval:.4f}")
    print(f"Heart Attack - Random Forest AUC: {heart_rf_eval:.4f}")
    
    if heart_rf_eval > heart_lr_eval:
        print("Random Forest performs better for heart attack prediction")
        best_heart_model = heart_rf_model
        best_heart_model_path = os.path.join(models_dir, "heart_model_rf")
    else:
        print("Logistic Regression performs better for heart attack prediction")
        best_heart_model = heart_lr_model
        best_heart_model_path = os.path.join(models_dir, "heart_model_lr")
    
    # Save links to best models
    with open(os.path.join(models_dir, "best_models.txt"), "w") as f:
        f.write(f"Best diabetes model: {best_diabetes_model_path}\n")
        f.write(f"Best heart model: {best_heart_model_path}\n")
    
    # Stop Spark session
    spark.stop()


if __name__ == "__main__":
    main() 