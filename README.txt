# CSE-6242-final

# USER GUIDE 
### DESCRIPTION 
We used various python data science and visualization packages for our analysis and output. The packages and their versions are captured in requirements.txt as part of this repository. 
After starting up the visualization, please wait for a few minutes until the backend compressing is complete in order for the visualization to render.
 

### INSTALLATION 
In order to view the visualization please run the following line in a terminal with anaconda already installed.
- Anaconda installation: https://www.anaconda.com/products/individual
- Create Anaconda environment: conda create -n viz_course python=3.8
- Activate the conda environment in the command line: conda activate viz_course
- Note: Please make sure you are in the root directory of this repository: cd CSE-6242-final
- Package installation: pip install -r requirements.txt
- Lastly, install prophet library using conda: conda install -c conda-forge fbprophet


### EXECUTION 
To see the visualization, please run the following command in the root directory of the repo:
- streamlit run streamlit_chart.py

Navigate to the network url in the terminal
Ex. http://192.168.102.78:8501/
 
### DEMO VIDEO 
https://youtu.be/WHn3OBNg8ps