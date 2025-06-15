import os
import subprocess
import sys
import time
import platform

def print_header(text):
    """Print a formatted header"""
    print("\n" + "="*80)
    print(f"{text.center(80)}")
    print("="*80 + "\n")

def run_command(command, description, shell=True):
    """Run a command with proper error handling and real-time output"""
    print_header(description)
    print(f"Running: {command}\n")
    start_time = time.time()
    
    try:
        # Special handling for Windows
        if platform.system() == "Windows" and not shell:
            # For non-shell commands on Windows, we need to handle path escaping differently
            # This is important if we ever switch to shell=False
            if isinstance(command, str):
                command = command.replace('"', '')  # Remove quotes that we may have added
            
        # Use appropriate shell based on OS
        process = subprocess.Popen(
            command,
            shell=shell,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding='utf-8',
            errors='replace'
        )
        
        # Print output in real-time
        for line in process.stdout:
            print(line.strip())
        
        process.wait()
        end_time = time.time()
        
        if process.returncode == 0:
            print(f"\n✅ {description} completed successfully in {end_time - start_time:.2f} seconds")
            return True
        else:
            print(f"\n❌ {description} failed with return code {process.returncode}")
            return False
            
    except Exception as e:
        print(f"\n❌ {description} failed with error: {str(e)}")
        return False

def setup_environment():
    """Set up environment variables and dependencies"""
    print_header("Setting Up Environment")
    
    # Create necessary directories
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    
    # Check Python version
    python_version = platform.python_version()
    print(f"Python version: {python_version}")
    
    # Check if running in virtual environment
    in_venv = hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)
    if in_venv:
        print("✅ Running in virtual environment")
    else:
        print("⚠️ Not running in virtual environment. Consider activating one.")
    
    # Check for required packages
    required_packages = ["pyspark", "streamlit", "pandas", "numpy", "matplotlib"]
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package} is installed")
        except ImportError:
            print(f"❌ {package} is not installed")
            missing_packages.append(package)
    
    if missing_packages:
        print("\n⚠️ Some required packages are missing. Installing them now...")
        requirements_cmd = f"{sys.executable} -m pip install -r requirements.txt"
        if not run_command(requirements_cmd, "Installing required packages"):
            print("❌ Failed to install required packages. Please install them manually.")
            print(f"   Command: {requirements_cmd}")
            return False
    
    # Set environment variables for Hadoop and Java
    if platform.system() == "Windows":
        # Check if HADOOP_HOME exists or set default
        if "HADOOP_HOME" not in os.environ:
            hadoop_path = r"C:\hadoop"
            if os.path.exists(hadoop_path):
                os.environ["HADOOP_HOME"] = hadoop_path
                print(f"✅ Set HADOOP_HOME to {hadoop_path}")
            else:
                print(f"⚠️ Hadoop directory not found at {hadoop_path}.")
                print("   You may need to set up Hadoop manually or adjust paths.")
        else:
            print(f"✅ HADOOP_HOME is set to {os.environ['HADOOP_HOME']}")
        
        # Check if JAVA_HOME exists
        if "JAVA_HOME" not in os.environ:
            # Try to detect JAVA_HOME
            possible_java_homes = [r"C:\JAVA\jdk8", r"C:\Program Files\Java\jdk-11", r"C:\Program Files\Java\jdk1.8.0"]
            for java_path in possible_java_homes:
                if os.path.exists(java_path):
                    os.environ["JAVA_HOME"] = java_path
                    print(f"✅ Set JAVA_HOME to {java_path}")
                    break
            else:
                print("⚠️ JAVA_HOME not set and couldn't detect Java installation path.")
                print("   You may need to set up Java manually.")
        else:
            print(f"✅ JAVA_HOME is set to {os.environ['JAVA_HOME']}")
    
    return True

def process_data():
    """Run data processing script"""
    # Always use absolute paths
    healthify_dir = os.path.abspath(os.getcwd())
    script_path = os.path.join(healthify_dir, "data_processing.py")
    
    print(f"Attempting to run script at: {script_path}")
    print(f"Current working directory: {os.getcwd()}")
    
    # Make sure the file exists before attempting to run it
    if not os.path.exists(script_path):
        print(f"❌ Error: Script not found at {script_path}")
        # Try the src directory as fallback
        script_path = os.path.join(healthify_dir, "src", "data_processing.py")
        if not os.path.exists(script_path):
            print(f"❌ Error: Script not found in src directory either")
            return False
        print(f"Found script in src directory: {script_path}")
    
    # Check if we're in the top-level HEALTHIFY directory
    # This is important for relative paths in the scripts
    cwd_basename = os.path.basename(os.getcwd())
    if cwd_basename != "HEALTHIFY":
        print(f"Current directory is {cwd_basename}, expected HEALTHIFY")
        print("Changing to the correct directory...")
        
        # Try to find and change to the HEALTHIFY directory
        if os.path.exists("HEALTHIFY"):
            os.chdir("HEALTHIFY")
            print(f"Changed to: {os.getcwd()}")
    
    # Quote paths to handle spaces in directory names
    # Use -u flag to ensure python output is unbuffered
    command = f'"{sys.executable}" -u "{script_path}"'
    return run_command(command, "Data Processing")

def train_models():
    """Run model training script"""
    # Always use absolute paths
    healthify_dir = os.path.abspath(os.getcwd())
    script_path = os.path.join(healthify_dir, "model_training.py")
    
    print(f"Attempting to run script at: {script_path}")
    
    # Make sure the file exists before attempting to run it
    if not os.path.exists(script_path):
        print(f"❌ Error: Script not found at {script_path}")
        # Try the src directory as fallback
        script_path = os.path.join(healthify_dir, "src", "model_training.py")
        if not os.path.exists(script_path):
            print(f"❌ Error: Script not found in src directory either")
            return False
        print(f"Found script in src directory: {script_path}")
    
    # Quote paths to handle spaces in directory names
    command = f'"{sys.executable}" "{script_path}"'
    return run_command(command, "Model Training")

def run_app():
    """Run the Streamlit app"""
    # Always use absolute paths
    healthify_dir = os.path.abspath(os.getcwd())
    script_path = os.path.join(healthify_dir, "app.py")
    
    print(f"Attempting to run script at: {script_path}")
    
    # Make sure the file exists before attempting to run it
    if not os.path.exists(script_path):
        print(f"❌ Error: Script not found at {script_path}")
        # Try the src directory as fallback
        script_path = os.path.join(healthify_dir, "src", "app.py")
        if not os.path.exists(script_path):
            print(f"❌ Error: Script not found in src directory either")
            return False
        print(f"Found script in src directory: {script_path}")
    
    # Quote paths to handle spaces in directory names
    command = f'streamlit run "{script_path}"'
    return run_command(command, "Starting Streamlit App")

def main():
    """Main function to run the HEALTHIFY pipeline"""
    print_header("HEALTHIFY - Health Prediction System")
    
    # Check if we're in the correct directory structure
    current_dir = os.getcwd()
    current_dir_name = os.path.basename(current_dir)
    
    print(f"Starting in directory: {current_dir}")
    
    # Try different strategies to find the HEALTHIFY directory
    if current_dir_name != "HEALTHIFY":
        # Option 1: Check if we're in the parent dir of HEALTHIFY
        healthify_dir = os.path.join(current_dir, "HEALTHIFY")
        if os.path.isdir(healthify_dir):
            print(f"Found HEALTHIFY directory at: {healthify_dir}")
            os.chdir(healthify_dir)
            print(f"Changed working directory to: {os.getcwd()}")
        else:
            # Option 2: Maybe we're in a subdirectory of HEALTHIFY
            parent_dir = os.path.dirname(current_dir)
            if os.path.basename(parent_dir) == "HEALTHIFY":
                os.chdir(parent_dir)
                print(f"Changed working directory to parent: {os.getcwd()}")
            else:
                print("⚠️ Warning: Could not locate HEALTHIFY directory")
                print("Current directory structure may not match expected project layout")
                print("Some operations might fail if paths aren't resolved correctly")
    
    # Setup environment and dependencies
    if not setup_environment():
        print("❌ Environment setup failed. Please fix the issues and try again.")
        return
    
    # Run pipeline steps with option to skip
    steps = [
        ("Process data", process_data),
        ("Train models", train_models),
        ("Run Streamlit app", run_app)
    ]
    
    for step_name, step_func in steps:
        choice = input(f"\nDo you want to {step_name.lower()}? (y/n): ").strip().lower()
        if choice == 'y':
            if not step_func():
                choice = input("\nStep failed. Continue anyway? (y/n): ").strip().lower()
                if choice != 'y':
                    print("Exiting pipeline.")
                    break
        else:
            print(f"Skipping: {step_name}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nProcess interrupted by user. Exiting...")
    except Exception as e:
        print(f"\n\nAn unexpected error occurred: {str(e)}")
        import traceback
        traceback.print_exc() 