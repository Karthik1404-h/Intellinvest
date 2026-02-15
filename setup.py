#!/usr/bin/env python3
"""
Setup script for Portfolio Optimizer project
Handles dependency installation with better error handling
"""

import subprocess
import sys
import os

def run_command(command, description):
    """Run a command and handle errors gracefully"""
    print(f"\n🔧 {description}")
    print(f"Running: {command}")
    
    try:
        result = subprocess.run(command, shell=True, check=True, 
                              capture_output=True, text=True)
        print(f"✅ Success: {description}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed: {description}")
        print(f"Error: {e}")
        if e.stdout:
            print(f"stdout: {e.stdout}")
        if e.stderr:
            print(f"stderr: {e.stderr}")
        return False

def check_package(package_name):
    """Check if a package is installed"""
    try:
        __import__(package_name)
        return True
    except ImportError:
        return False

def main():
    """Main setup function"""
    print("🚀 Portfolio Optimizer Setup")
    print("=" * 50)
    
    # Step 1: Update pip and core tools
    print("\n📦 Step 1: Updating package managers...")
    run_command("python -m pip install --upgrade pip setuptools wheel", 
                "Updating pip, setuptools, and wheel")
    
    # Step 2: Install core dependencies
    print("\n📦 Step 2: Installing core dependencies...")
    core_packages = [
        "numpy>=1.21.0",
        "pandas>=1.5.0", 
        "matplotlib>=3.6.0",
        "seaborn>=0.11.0",
        "scikit-learn>=1.2.0",
        "yfinance>=0.2.18",
        "loguru>=0.7.0",
        "tqdm>=4.64.0",
        "jupyter>=1.0.0"
    ]
    
    for package in core_packages:
        if not run_command(f"pip install {package}", f"Installing {package}"):
            print(f"⚠️  Warning: Failed to install {package}")
    
    # Step 3: Install optional dependencies
    print("\n📦 Step 3: Installing optional dependencies...")
    optional_packages = [
        "pypfopt>=1.5.0",
        "cvxpy>=1.2.0", 
        "hdbscan>=0.8.28",
        "pandas-ta>=0.3.14b",
        "statsmodels>=0.13.0",
        "plotly>=5.10.0"
    ]
    
    for package in optional_packages:
        run_command(f"pip install {package}", f"Installing {package}")
    
    # Step 4: Test imports
    print("\n🧪 Step 4: Testing package imports...")
    
    test_packages = {
        'numpy': 'numpy',
        'pandas': 'pandas', 
        'matplotlib': 'matplotlib.pyplot',
        'seaborn': 'seaborn',
        'sklearn': 'scikit-learn',
        'yfinance': 'yfinance',
        'loguru': 'loguru'
    }
    
    failed_imports = []
    
    for import_name, package_name in test_packages.items():
        if check_package(import_name):
            print(f"✅ {package_name} imported successfully")
        else:
            print(f"❌ {package_name} import failed")
            failed_imports.append(package_name)
    
    # Step 5: Test main script
    print("\n🧪 Step 5: Testing main script...")
    if run_command("python main.py --info", "Testing main script"):
        print("✅ Setup completed successfully!")
    else:
        print("❌ Setup incomplete. Please check errors above.")
    
    # Final recommendations
    print("\n💡 Setup Summary:")
    if failed_imports:
        print("⚠️  Some packages failed to install:")
        for pkg in failed_imports:
            print(f"   - {pkg}")
        print("\n📋 Manual installation commands:")
        print("   conda install numpy pandas matplotlib seaborn scikit-learn jupyter")
        print("   pip install yfinance loguru tqdm pypfopt")
    else:
        print("🎉 All core packages installed successfully!")
        
    print("\n🚀 Next steps:")
    print("   1. Run: python main.py --info")
    print("   2. Run: python main.py --full")
    print("   3. Open: notebooks/01_data_exploration.ipynb")

if __name__ == "__main__":
    main()