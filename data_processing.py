from pyspark.sql import SparkSession
from pyspark.sql.functions import col, mean, when
from pyspark.ml.linalg import VectorUDT, Vectors
from pyspark.sql.functions import udf
from pyspark.ml.feature import VectorAssembler, StandardScaler
import os
import sys
import subprocess

# Check if we have pandas installed for fallback
try:
    import pandas as pd
    import numpy as np
    from sklearn.preprocessing import StandardScaler as SklearnStandardScaler
    PANDAS_AVAILABLE = True
    print("Pandas and scikit-learn are available for fallback processing")
except ImportError:
    PANDAS_AVAILABLE = False
    print("Warning: Pandas/scikit-learn not available, no fallback option if Spark fails")

# Get the project root directory (current directory)
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
print(f"Project root directory: {PROJECT_ROOT}")

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
        
        # Disable Hadoop native libraries warning
        os.environ['HADOOP_OPTS'] = "-Djava.library.path=" + hadoop_bin
    else:
        print(f"Warning: Hadoop bin directory not found at {hadoop_bin}")
else:
    print(f"Warning: Hadoop directory not found at {hadoop_path}")
    print("Continuing without Hadoop configuration...")

# Try known Java locations
java_paths = [r'C:\JAVA\jdk8', r'C:\Program Files\Java\jdk-11', r'C:\Program Files\Java\jdk1.8.0']
java_found = False

for java_path in java_paths:
    if os.path.exists(java_path):
        os.environ['JAVA_HOME'] = java_path
        print(f"Found Java at: {java_path}")
        java_found = True
        break

if not java_found:
    print("Warning: No Java installation found in common locations")

# Verify Java is accessible
try:
    print("Checking Java availability...")
    # Don't exit if Java isn't available, just warn
    result = subprocess.run(['java', '-version'], capture_output=True, text=True)
    java_output = result.stderr if result.stderr else result.stdout
    print("Java detected:", java_output.split('\n')[0])
except Exception as e:
    print(f"Warning: Error checking Java: {e}")
    print("Continuing without verified Java - this may cause PySpark issues")

def initialize_spark():
    """Initialize Spark session with robust error handling"""
    try:
        print("Initializing Spark session...")
        # Add more configurations to help debug issues
        spark = SparkSession.builder \
            .appName("HEALTHIFY - Data Processing") \
            .config("spark.driver.memory", "2g") \
            .config("spark.driver.extraJavaOptions", "-XX:+PrintGCDetails -XX:+PrintGCTimeStamps") \
            .config("spark.python.worker.reuse", "false") \
            .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
            .getOrCreate()
            
        # Test that Spark can perform basic operations
        print("Testing Spark session...")
        test_df = spark.createDataFrame([(1, "test")], ["id", "value"])
        test_df.show()
        print("Spark is working correctly!")
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

def load_diabetes_data(spark):
    """Load diabetes dataset from CSV"""
    print("Loading diabetes dataset...")
    data_path = os.path.join(PROJECT_ROOT, "data", "diabetes.csv")
    print(f"Looking for diabetes data at: {data_path}")
    
    if not os.path.exists(data_path):
        print(f"Error: Diabetes dataset not found at {data_path}")
        print(f"Current working directory: {os.getcwd()}")
        print(f"Directory contents: {os.listdir(os.path.join(PROJECT_ROOT, 'data'))}")
        sys.exit(1)
        
    return spark.read.option("header", "true") \
        .option("inferSchema", "true") \
        .csv(data_path)

def load_heart_data(spark):
    """Load heart attack dataset from CSV"""
    print("Loading heart attack dataset...")
    data_path = os.path.join(PROJECT_ROOT, "data", "heart_attack_prediction_india.csv")
    print(f"Looking for heart attack data at: {data_path}")
    
    if not os.path.exists(data_path):
        print(f"Error: Heart attack dataset not found at {data_path}")
        print(f"Current working directory: {os.getcwd()}")
        print(f"Directory contents: {os.listdir(os.path.join(PROJECT_ROOT, 'data'))}")
        sys.exit(1)
        
    return spark.read.option("header", "true") \
        .option("inferSchema", "true") \
        .csv(data_path)

def process_diabetes_data(df):
    """Process diabetes dataset"""
    print("Processing diabetes dataset...")
    
    # Display basic statistics
    print(f"Total records: {df.count()}")
    print("Schema:")
    df.printSchema()
    
    # Replace zeros with column means for specific columns
    zero_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    
    for column in zero_cols:
        # Calculate mean excluding zeros
        mean_val = df.filter(col(column) != 0).agg(mean(column)).collect()[0][0]
        print(f"Mean {column}: {mean_val}")
        
        # Replace zeros with mean
        df = df.withColumn(column, when(col(column) == 0, mean_val).otherwise(col(column)))
    
    # Feature engineering
    feature_cols = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
                   'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
    
    # Create feature vector
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features_unscaled")
    df = assembler.transform(df)
    
    # Standardize features
    scaler = StandardScaler(inputCol="features_unscaled", outputCol="features", 
                           withStd=True, withMean=True)
    scaler_model = scaler.fit(df)
    df = scaler_model.transform(df)
    
    # Select relevant columns
    df = df.select("features", "Outcome")
    
    return df, scaler_model

def process_heart_data(df):
    """Process heart attack dataset"""
    print("Processing heart attack dataset...")
    
    # Display basic statistics
    print(f"Total records: {df.count()}")
    print("Schema:")
    df.printSchema()
    
    # Drop unnecessary columns
    drop_cols = ['Patient_ID', 'State_Name']
    df = df.drop(*drop_cols)
    
    # Handle missing values
    df = df.na.drop()
    
    # Convert Gender to numeric (1 for Male, 0 for Female)
    df = df.withColumn("Gender", when(col("Gender") == "Male", 1).otherwise(0))
    
    # Feature engineering
    feature_cols = [c for c in df.columns if c != 'Heart_Attack_Risk']
    
    # Create feature vector
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features_unscaled")
    df = assembler.transform(df)
    
    # Standardize features
    scaler = StandardScaler(inputCol="features_unscaled", outputCol="features", 
                           withStd=True, withMean=True)
    scaler_model = scaler.fit(df)
    df = scaler_model.transform(df)
    
    # Select relevant columns
    df = df.select("features", "Heart_Attack_Risk")
    
    return df, scaler_model

def save_processed_data(df, name):
    """Save processed data as parquet files to preserve Vector types"""
    output_path = os.path.join(PROJECT_ROOT, "data", "processed", name)
    print(f"Saving processed data to {output_path}")
    
    # Ensure the directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save as parquet to preserve the Vector type
    df.write.mode("overwrite").parquet(output_path)

def process_with_pandas():
    """Process data using pandas as a fallback when Spark fails"""
    print("\n==================================")
    print("RUNNING FALLBACK PANDAS PROCESSING")
    print("==================================\n")
    
    if not PANDAS_AVAILABLE:
        print("Error: Pandas not available for fallback processing")
        sys.exit(1)
        
    # Create necessary directories
    processed_dir = os.path.join(PROJECT_ROOT, "data", "processed")
    models_dir = os.path.join(PROJECT_ROOT, "models")
    os.makedirs(processed_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    
    # Process diabetes data
    diabetes_path = os.path.join(PROJECT_ROOT, "data", "diabetes.csv")
    if not os.path.exists(diabetes_path):
        print(f"Error: Diabetes dataset not found at {diabetes_path}")
        return
        
    print(f"Loading diabetes data from {diabetes_path}")
    diabetes_df = pd.read_csv(diabetes_path)
    print(f"Loaded {len(diabetes_df)} diabetes records")
    
    # Process heart data
    heart_path = os.path.join(PROJECT_ROOT, "data", "heart_attack_prediction_india.csv")
    if not os.path.exists(heart_path):
        print(f"Error: Heart dataset not found at {heart_path}")
        return
        
    print(f"Loading heart data from {heart_path}")
    heart_df = pd.read_csv(heart_path)
    print(f"Loaded {len(heart_df)} heart records")
    
    # Process diabetes data
    print("Processing diabetes data...")
    # Replace zeros with means
    zero_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    for col in zero_cols:
        mean_val = diabetes_df.loc[diabetes_df[col] != 0, col].mean()
        diabetes_df.loc[diabetes_df[col] == 0, col] = mean_val
    
    # Standardize features
    feature_cols = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
                   'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
    X = diabetes_df[feature_cols]
    y = diabetes_df['Outcome']
    
    # Save processed data
    diabetes_processed = os.path.join(processed_dir, "diabetes_pandas.csv")
    diabetes_df.to_csv(diabetes_processed, index=False)
    print(f"Saved processed diabetes data to {diabetes_processed}")
    
    # Process heart data
    print("Processing heart data...")
    # Drop unnecessary columns and handle missing values
    if 'Patient_ID' in heart_df.columns:
        heart_df = heart_df.drop(columns=['Patient_ID'])
    if 'State_Name' in heart_df.columns:
        heart_df = heart_df.drop(columns=['State_Name'])
    
    # Handle missing values
    heart_df = heart_df.dropna()
    
    # Convert gender to numeric
    if 'Gender' in heart_df.columns:
        heart_df['Gender'] = heart_df['Gender'].map({'Male': 1, 'Female': 0})
    
    # Save processed data
    heart_processed = os.path.join(processed_dir, "heart_pandas.csv")
    heart_df.to_csv(heart_processed, index=False)
    print(f"Saved processed heart data to {heart_processed}")
    
    print("\nData processing with pandas completed successfully.")
    return True

def main():
    """Main function to process both datasets"""
    try:
        # Initialize Spark
        spark = initialize_spark()
        
        # Create directories if they don't exist
        os.makedirs(os.path.join(PROJECT_ROOT, "data", "processed"), exist_ok=True)
        os.makedirs(os.path.join(PROJECT_ROOT, "models"), exist_ok=True)
        
        print(f"Working directory: {os.getcwd()}")
        print(f"Project root: {PROJECT_ROOT}")
        
        # Process diabetes data
        diabetes_df = load_diabetes_data(spark)
        processed_diabetes, diabetes_scaler = process_diabetes_data(diabetes_df)
        save_processed_data(processed_diabetes, "diabetes")
        
        # Process heart attack data
        heart_df = load_heart_data(spark)
        processed_heart, heart_scaler = process_heart_data(heart_df)
        save_processed_data(processed_heart, "heart")
        
        # Show sample of processed data
        print("\nSample of processed diabetes data:")
        processed_diabetes.show(5)
        
        print("\nSample of processed heart attack data:")
        processed_heart.show(5)
        
        # Stop Spark session
        spark.stop()
        
    except Exception as e:
        print(f"Error in Spark processing: {e}")
        print("Attempting fallback to pandas processing...")
        process_with_pandas()

if __name__ == "__main__":
    main() 