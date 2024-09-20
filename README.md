# Climate Change: Surface Temperature Prediction and Visualization
--------------------------------------------------------------------------------

A virtual environment is used to run all the code and corresponding visualization. 
Please follow the intructions below to setup and run the code. 

1. Navigate to the code/ directory in the terminal. Setup a python virtual environment using the following command. <br />
   pip -m venv env
2. Then activate the environment with the given requirements in the `requirements.txt` file based on os. <br />
   pip install -r requirements.txt
3. Download the dataset files from the given kaggle link and put them in the code/data/source/ directory. <br />
    https://www.kaggle.com/datasets/berkeleyearth/climate-change-earth-surface-temperature-data  
4. Clean the dataset and bring it to an appropriate format using the `data_clean.py` file and the following command. <br />
    python data_clean.py
5. Train the models and generate the predictions using the most suitable model. <br />
    To do this, open the `prediction_models.ipynb` file in jupyter notebook or jupyter lab and run all the sections. 

You can alternatively download the prediction data from our trained Random Forest model on this website: 
https://zenodo.org/record/7834328

There is a separate .csv file for each year, which will aid in loading the appropriate file for the selected year in the visualization code.

Several other models have been trained and tested, apart from the final Random Forest, which can be found in the `prediction_models.ipynb` file or 
individually in the various .py files present in the code/models/ directory. 

6. Run the visualization. This can be done by going inside the code directory and running a python server (python -m http.server 8000) to serve files from there. <br />
    After you have the python server running, go to localhost:8000 and navigate to the `climate_visualization.html` file in the visualization folder which will allow you to interact with the tool. 
