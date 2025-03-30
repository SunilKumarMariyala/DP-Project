# Solar Panel Fault Detection System

A machine learning-based system for detecting faults in solar panels using electrical measurements. The system achieves 96.25% testing accuracy in identifying healthy panels and various fault conditions.

## Project Overview

This project implements a comprehensive solar panel fault detection system that can:
- Detect 5 different panel states (Healthy + 4 fault types)
- Provide real-time monitoring and alerts
- Visualize performance data through an interactive dashboard
- Generate predictions with just two simple measurements (PV Current and Voltage)

## Features

- **High Accuracy**: 96.25% accuracy in testing
- **Simple Input**: Only requires PV Current and Voltage measurements
- **Real-time Monitoring**: Continuous monitoring with automatic alerts
- **Detailed Analysis**: Provides confidence scores and recommended actions
- **MATLAB Integration**: Optional connection to MATLAB for advanced simulations

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

### Step 5: Initialize Git in Your Project Folder

Open Command Prompt and navigate to your project folder:

```bash
cd path\to\DP-Project
```

Initialize Git in this folder:

```bash
git init
```

### Step 6: Add Files to Git

Add all files to be tracked by Git:

```bash
git add .
```

### Step 7: Create .gitignore File (Optional but Recommended)

Create a file named `.gitignore` in your project folder with the following content:

```
# Python cache files
__pycache__/
*.py[cod]
*$py.class

# Virtual environment
venv/
env/
ENV/

# Database files (if you don't want to include sample data)
# *.db

# Log files
*.log

# MATLAB generated files
matlab_data/

# IDE specific files
.idea/
.vscode/

# Distribution / packaging
dist/
build/
*.egg-info/
```

### Step 8: Commit Your Changes

Commit the added files with a message:

```bash
git commit -m "Initial commit: Solar Panel Fault Detection System"
```

### Step 9: Link to Your GitHub Repository

Connect your local repository to the GitHub repository:

```bash
git remote add origin https://github.com/YourUsername/Solar-Panel-Fault-Detection.git
```

Replace `YourUsername` with your actual GitHub username and `Solar-Panel-Fault-Detection` with your repository name.

### Step 10: Push Your Code to GitHub

Push your code to GitHub:

```bash
git push -u origin master
```

Note: If your default branch is named "main" instead of "master", use:

```bash
git push -u origin main
```

### Step 11: Verify Your Upload

1. Go to your GitHub repository page
2. Refresh the page if needed
3. You should see all your project files listed

## Best Practices for GitHub

1. **Commit Often**: Make small, focused commits with clear messages
2. **Use Branches**: Create branches for new features or bug fixes
3. **Write Good Commit Messages**: Be clear and descriptive
4. **Keep Sensitive Data Out**: Don't commit API keys, passwords, or personal data
5. **Update README**: Keep your README updated as your project evolves

## Collaborating with Others

1. **Pull Requests**: Use pull requests for code reviews and collaboration
2. **Issues**: Use GitHub Issues to track bugs and feature requests
3. **Projects**: Use GitHub Projects for project management
4. **Actions**: Set up GitHub Actions for continuous integration/deployment

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

If you have any questions or feedback, please open an issue on GitHub or contact the repository owner.

Happy coding!
