# Climate change: Surface temperature prediction and visualization
--------------------------------------------------------------------------------

We use a virtual environment to run all the code and corresponding visualization. 
Please follow the intructions below to setup and run the code. 

1. To setup the environment
pip -m venv env
Then activate the envornoment based on os
pip install -r requirements.txt 

2. The dataset files from the source are provided under the code/data/original/ directory. 

3. Clean the dataset and bring it to an appropriate format using the following command: 
python data_clean.py 

4. Train the model and generate the predictions. 
To do this, Open the file predictions-rf.ipynb in jupyter notebook or jupyter lab and run all the sections. 

Alternatively, the prediction data from our trained model is provided under the code/data/prediction/ directory: 

There is a separate .csv file for each year, which will aid in loading the appropriate file for the selected year in the visualization code.

We have trained and tested a bunch of other models also apart from final Random Forest, 
which can be found in the various .ipynb files present in the code directory. 

5. Run the visualization. This can be done by going inside the code directory 
and running a python server (python -m http.server 8000) to serve files from there. After you have the python server running, 
go to localhost:8000 and navigate to the climate_visualization.html file in the visualization folder will allow you to interact with the tool. 
