import subprocess
import sys
import os
import platform
import venv
from pathlib import Path
import shutil

def create_venv():
    venv_dir = Path("venv")
    if venv_dir.exists():
        print("Removing existing virtual environment...")
        shutil.rmtree(venv_dir)
    print("Creating new virtual environment...")
    venv.create(venv_dir, with_pip=True)

def get_pip_path():
    if platform.system() == "Windows":
        return Path("venv/Scripts/pip.exe")
    return Path("venv/bin/pip")

def get_python_path():
    if platform.system() == "Windows":
        return Path("venv/Scripts/python.exe")
    return Path("venv/bin/python")

def install_requirements():
    pip_path = get_pip_path()
    print("Installing requirements...")

    subprocess.run([str(pip_path), "install", "--upgrade", "pip"], check=True)
    
    # Install requirements
    result = subprocess.run([str(pip_path), "install", "-r", "requirements.txt"], check=True)
    if result.returncode != 0:
        raise Exception("Error installing requirements")

    print("\nVerifying installation...")
    subprocess.run([str(pip_path), "list"], check=True)

def run_streamlit():
    python_path = get_python_path()
    print("\nRunning Streamlit app...")
    print("Press Ctrl+C to stop the app")
    
    try:
        # Start Streamlit in a subprocess with direct terminal output
        process = subprocess.Popen(
            [str(python_path), "-m", "streamlit", "run", "app.py"],
            stdout=None,
            stderr=None,
            stdin=subprocess.DEVNULL
        )

        process.wait()
        
    except KeyboardInterrupt:
        print("\nStopping Streamlit app...")
        try:
            process.terminate()
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            print("Force stopping Streamlit app...")
            process.kill()
        print("Streamlit app stopped successfully.")
        sys.exit(0)
    except Exception as e:
        print(f"Error running Streamlit app: {e}")
        sys.exit(1)

def check_python_version():
    major, minor, micro = sys.version_info[:3]
    if major != 3 or minor != 13 or micro < 1:
        print("Error: This script requires Python 3.13.1 or higher.")
        print(f"Current Python version: {sys.version}")
        return False
    print(f"Using Python {sys.version}")
    return True

def main():
    print("Setting up the environment...")

    if not check_python_version():
        sys.exit(1)
    
    try:
        create_venv()
        install_requirements()
        run_streamlit()
    except KeyboardInterrupt:
        print("\nSetup interrupted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 
