# Quennect

0. Download Python (if you don't have one)

1. Download Visual Studio Code
   * Based on my years at Penn, I figured Visual Studio Code is the best tool to use especially when we are going to use SSH
   * After downloading VSCode, add the following extensions
     1. ms-vscode-remote.remote-ssh
     2. ms-python.python
     3. donjayamanne.python-extension-pack
     4. ms-toolsai.jupyter
     5. This sth extra, but consider getting Github Copilot (It is free for students! YAY) because it is extremely useful and cuts down your coding time to half.
    
2. Download Anaconda (if you don't have one)
   * Check out this link to the installation guide (https://docs.anaconda.com/free/anaconda/install/index.html)
   * Miniconda should also be fine
  
3. Clone this repo
   * If you are not familiar with Linux stuff and if you use Windows, I recommend downloading Git Bash(https://gitforwindows.org/). The other alternative is setting up WSL(Window Subsystem for Linux), which is pretty extra but this brings greater peace and joy in your life.
   * Go to the directory where you want to make a clone for the repo:
       `git clone https://github.com/lsebin/Quennect.git`

4. Make a Python environment
   * I figured it is easier for us to open or work on any file when we have the same version of Python libraries. So, we will use a Python environment.
   `conda create -n qnct python=3.8`
   `conda activate qnct`
   `pip install -r requirements.txt`
   
   To use this virtual environment,
   `conda activate qnct`
