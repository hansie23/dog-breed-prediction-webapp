# Dog Breed Prediction

### Requirements -
Python  
Flask  
Numpy  
Tensorflow / Tensorflow-CPU

### Structure -
* **app.py** - This file contains the Flask APIs that load the initial screen of the web app, takes input image from the user, predicts using the loaded model and returns the result.
* **templates/web.html** - This is the primary HTML template.
* **static/style.css** - This is the main CSS file for styling the HTML template.
* **models/model1/** - This folder contains the model that was trained and saved, and now loaded in the 'app.py' file for predictions.
* **requirements.txt** - Contains all the required packages for the program.
* **Dockerfile** - Contains code to create a Docker container.



### Deploy on local machine -
- Make sure you are in the project home directory. Execute the below command in CMD:  

    *`python app.py`* 

- Open the URL in a browser.<br><br>

