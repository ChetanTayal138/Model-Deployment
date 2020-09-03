# Introduction 

Hello and welcome to this class on model deployment! We will be going through the process of understanding how model deployment works, basics of container technology, how models are served in production environments and help you deploy your first deep learning web application. Below are some pre-requisites for you to go through that will help us conduct the class in a smooth manner. Make sure to go through the steps and setup your machine. Happy Coding!

# Pre-Requisites 

First things first, lets help you create an isolated environment with all of the stuff that will be needed by your machine to run the code. This will prevent conflicts between different dependencies between different projects on your machine, letting you create projects in a controlled manner. My preferred way of managing different environments when creating deep learning projects is to use conda. So lets install that using the steps below :-

1. Install anaconda from https://docs.anaconda.com/anaconda/install/ 

    1. Windows : https://docs.anaconda.com/anaconda/install/windows/
    2. Mac : https://docs.anaconda.com/anaconda/install/mac-os/
    3. Linux : https://docs.anaconda.com/anaconda/install/linux/

2. Check the version of your conda version to make sure it was properly installed.

            conda --version

3. Create a new conda environment with python-3.7, flask and tensorflow using the following command. 

            conda create -n 'your-env-name' python=3.7 tensorflow flask 

4. Activate the conda environment using 
    
            conda activate 'your-env-name'

5. Run the python shell and import tensorflow. If you are able to import tensorflow without any problems, that means everything is working perfectly and we are good to go. 

           ![verification_image](https://github.com/ChetanTayal138/Model-Deployment/images/verify_tensorflow.png)



            




