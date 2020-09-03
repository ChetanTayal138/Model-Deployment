# Introduction 

Hello and welcome to this class on model deployment! We will be going through the process of understanding how model deployment works, basics of container technology, how models are served in production environments and help you deploy your first deep learning web application. Below are some pre-requisites for you to go through that will help us conduct the class in a smooth manner. Make sure to go through the steps and setup your machine. Happy Coding!

## Pre-Requisites 


### 1. Anaconda 

First things first, lets help you create an isolated environment with all project requirements. This will prevent conflicts between different dependencies between different projects on your machine, letting you create projects in a controlled manner. My preferred environment manager for deep learning projects is conda.

1. Install anaconda from https://docs.anaconda.com/anaconda/install/ 

    a) Windows : https://docs.anaconda.com/anaconda/install/windows/
    
    b) Mac : https://docs.anaconda.com/anaconda/install/mac-os/
    
    c) Linux : https://docs.anaconda.com/anaconda/install/linux/

2. Check the version of your conda version to make sure it was properly installed.

                    conda --version

3. Create a new conda environment with python-3.7, flask and tensorflow using the following command. 

                    conda create -n 'your-env-name' python=3.7 tensorflow flask 

4. Activate the conda environment using 
            
                    conda activate 'your-env-name'

5. Run the python shell and import tensorflow. If you are able to import tensorflow without any problems, that means everything is working perfectly and we are good to go. 

![verification_image](https://github.com/ChetanTayal138/Model-Deployment/blob/master/images/verify_tensorflow.png)


### 2. Git 

Git is a version controlling tool that helps track changes in code, collaborative development and many other advantages. For now we need git just for the sake of getting this repo on your machine. For now, we need git only for cloning the repository to your machine.

    1. Install git from https://git-scm.com/downloads.

    2. Clone this repository to your machine using
            
            git clone https://github.com/ChetanTayal138/Model-Deployment.git



    








            




