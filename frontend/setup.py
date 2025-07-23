import subprocess
import sys
import os

def install_requirements():
    """Install required packages"""
    print("Installing required packages...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ All packages installed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing packages: {e}")
        return False
    return True

def setup_secrets():
    """Setup secrets file"""
    secrets_dir = ".streamlit"
    secrets_file = os.path.join(secrets_dir, "secrets.toml")
    example_file = os.path.join(secrets_dir, "secrets.toml.example")
    
    # Create .streamlit directory if it doesn't exist
    if not os.path.exists(secrets_dir):
        os.makedirs(secrets_dir)
        print(f"✅ Created {secrets_dir} directory")
    
    # Check if secrets file exists
    if not os.path.exists(secrets_file):
        if os.path.exists(example_file):
            print(f"📝 Please copy {example_file} to {secrets_file} and add your API keys")
        else:
            print(f"📝 Please create {secrets_file} with your API keys")
        return False
    else:
        print("✅ Secrets file found")
        return True

def main():
    print("🚀 Setting up CV Analyzer App...")
    
    # Install requirements
    if not install_requirements():
        return
    
    # Setup secrets
    setup_secrets()
    
    print("\n🎉 Setup complete!")
    print("\nTo run the app:")
    print("1. Add your API keys to .streamlit/secrets.toml")
    print("2. Run: streamlit run scripts/app.py")

if __name__ == "__main__":
    main()
