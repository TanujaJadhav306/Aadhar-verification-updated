"""
Setup script for proctoring service
Helps with initial setup and model downloads
"""

import os
import sys
import subprocess


def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 9):
        print("Error: Python 3.9 or higher is required")
        sys.exit(1)
    print(f"✓ Python {sys.version_info.major}.{sys.version_info.minor} detected")


def install_dependencies():
    """Install required dependencies"""
    print("\nInstalling dependencies...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✓ Dependencies installed successfully")
    except subprocess.CalledProcessError:
        print("✗ Failed to install dependencies")
        sys.exit(1)


def create_env_file():
    """Create .env file from .env.example if it doesn't exist"""
    if not os.path.exists(".env"):
        if os.path.exists(".env.example"):
            import shutil
            shutil.copy(".env.example", ".env")
            print("✓ Created .env file from .env.example")
        else:
            print("⚠ .env.example not found, skipping .env creation")
    else:
        print("✓ .env file already exists")


def create_models_directory():
    """Create models directory"""
    os.makedirs("models", exist_ok=True)
    print("✓ Models directory created")


def main():
    """Main setup function"""
    print("=" * 50)
    print("AI Proctoring Service - Setup")
    print("=" * 50)
    
    check_python_version()
    create_models_directory()
    create_env_file()
    
    # Ask user if they want to install dependencies
    response = input("\nInstall dependencies? (y/n): ").strip().lower()
    if response == 'y':
        install_dependencies()
    else:
        print("Skipping dependency installation")
        print("Run: pip install -r requirements.txt")
    
    print("\n" + "=" * 50)
    print("Setup complete!")
    print("=" * 50)
    print("\nNext steps:")
    print("1. Review and update .env file with your settings")
    print("2. Run: python main.py")
    print("3. Access API docs at: http://localhost:8000/docs")


if __name__ == "__main__":
    main()
