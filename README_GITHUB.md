# Solar Panel Fault Detection System

A machine learning-based system for detecting faults in solar panels using electrical measurements. The system achieves 96.25% testing accuracy in identifying healthy panels and various fault conditions.

## Project Overview

This project implements a comprehensive solar panel fault detection system that can:
- Detect 5 different panel states (Healthy + 4 fault types)
- Provide real-time monitoring and alerts
- Visualize performance data through an interactive dashboard
- Generate predictions with just two simple measurements (PV Current and Voltage)
- Integrate with MATLAB for continuous data flow and simulation
- Store and analyze time-series data using MySQL

## Features

- **High Accuracy**: 96.25% accuracy in testing
- **Simple Input**: Only requires PV Current and Voltage measurements
- **Real-time Monitoring**: Continuous monitoring with automatic alerts
- **Detailed Analysis**: Provides confidence scores and recommended actions
- **MATLAB Integration**: Connection to MATLAB for advanced simulations
- **MySQL Database**: Efficient storage and retrieval of solar panel data
- **Web Dashboard**: Interactive visualization of panel performance
- **REST API**: Programmatic access to system functionality

## Quick Start

### Prerequisites
- Python 3.8+
- MySQL 5.7+
- MATLAB R2019b+ (optional, for MATLAB integration)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/Solar-Panel-Fault-Detection.git
   cd Solar-Panel-Fault-Detection
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up the MySQL database:
   ```bash
   python database_setup.py
   ```

4. Set environment variables:
   ```bash
   # For Windows PowerShell
   $env:DB_HOST = "localhost"
   $env:DB_USER = "solar_user"
   $env:DB_PASSWORD = "your_secure_password"
   $env:DB_NAME = "solar_panel_db"
   
   # For MATLAB integration (optional)
   $env:MATLAB_PATH = "C:\Program Files\MATLAB\R2023b\bin\matlab.exe"
   $env:MATLAB_MODEL_PATH = "path\to\your\GridConnectedPVFarmExample"
   ```

5. Run the application:
   ```bash
   python app.py
   ```

6. Open a web browser and navigate to:
   ```
   http://127.0.0.1:5000
   ```

## System Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│                 │     │                 │     │                 │
│  Solar Panels   │────▶│  MATLAB Model   │────▶│  Data Processor │
│                 │     │                 │     │                 │
└─────────────────┘     └─────────────────┘     └────────┬────────┘
                                                         │
                                                         ▼
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│                 │     │                 │     │                 │
│   Web Dashboard │◀────│  ML Prediction  │◀────│  MySQL Database │
│                 │     │                 │     │                 │
└─────────────────┘     └─────────────────┘     └─────────────────┘
```

## Fault Types Detected

1. **Healthy Panel**: Normal operation
2. **Line-Line Fault**: Two points in the array are connected
3. **Open Circuit Fault**: Circuit is broken somewhere in the array
4. **Partial Shading**: Some panels are receiving less sunlight
5. **Panel Degradation**: Efficiency is lower than expected

## Screenshots

![Dashboard](./static/img/dashboard.png)
![Prediction](./static/img/prediction.png)
![MATLAB Integration](./static/img/matlab.png)

## API Reference

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/predict` | POST | Make a prediction based on PV current and voltage |
| `/api/status` | GET | Get system status and performance metrics |
| `/api/history` | GET | Get historical prediction data |
| `/api/start` | POST | Start real-time monitoring |
| `/api/stop` | POST | Stop real-time monitoring |

## Technologies Used

- **Backend**: Python, Flask, SQLAlchemy
- **Frontend**: HTML, CSS, JavaScript, Chart.js
- **Database**: MySQL
- **Machine Learning**: PyTorch, scikit-learn
- **Simulation**: MATLAB Engine for Python

## How to Upload This Project to GitHub

### Step 1: Create a GitHub Account

If you don't already have one, create a GitHub account at [github.com](https://github.com/).

### Step 2: Install Git

1. Download Git from [git-scm.com](https://git-scm.com/downloads)
2. Install Git following the installation instructions
3. Verify installation by opening Command Prompt and typing `git --version`

### Step 3: Configure Git

Open Command Prompt and set up your identity:

```bash
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"
```

### Step 4: Create a New Repository on GitHub

1. Log in to GitHub
2. Click the "+" icon in the top right corner and select "New repository"
3. Name your repository (e.g., "Solar-Panel-Fault-Detection")
4. Add a description (optional)
5. Choose whether to make it public or private
6. Do NOT initialize with README, .gitignore, or license (we'll push our existing project)
7. Click "Create repository"

### Step 5: Initialize Git in Your Project

Navigate to your project directory in Command Prompt:

```bash
cd path\to\your\project
git init
```

### Step 6: Create .gitignore File

Create a `.gitignore` file in your project directory to exclude unnecessary files:

```
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
*.egg-info/
.installed.cfg
*.egg

# Virtual Environment
venv/
ENV/

# IDE
.idea/
.vscode/
*.swp
*.swo

# Database
*.db
*.sqlite3

# Environment variables
.env

# MATLAB
*.asv
*.mex*
slprj/
```

### Step 7: Add Files to Git

Add your project files to Git:

```bash
git add .
```

### Step 8: Commit Changes

Commit your files with a message:

```bash
git commit -m "Initial commit"
```

### Step 9: Add Remote Repository

Add the GitHub repository as a remote:

```bash
git remote add origin https://github.com/yourusername/Solar-Panel-Fault-Detection.git
```

### Step 10: Push to GitHub

Push your code to GitHub:

```bash
git push -u origin master
```

You may need to authenticate with your GitHub username and password or token.

### Step 11: Verify Upload

Go to your GitHub repository page to verify that your files have been uploaded successfully.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- The MATLAB GridConnectedPVFarm model for solar panel simulation
- PyTorch for the machine learning framework
- Flask for the web interface
