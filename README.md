After downloading the files from github or using git clone please ensure your directory structure looks like this
```
Directory Structure:-
main_folder
	|
	| -------- image_classification
	|		|
	| 		| —------- app.py
	|		|
	| 		| —------- best_model.pt
	| 		|
	| 		| —------- frontend.py
	|		|
	| 		| —------- image_classification.ipynb
	|		|
	| 		| —------- Image_Classification_Report.docx
	|		|
	| 		|—------- inference.py
	|		|
	| 		| —------- README.txt
```
## On windows please use command prompt
It is recommended to create a new virtual environment<br>
```python -m venv image_classification``` (Windows)<br>
```python3 -m venv image_classification``` ( MacOS, linux)<br>

```image_classification\Scripts\activate``` (Windows)<br>
```source image_classification/bin/activate``` (MacOS, linux)<br>

# Libraries required for image_classification: 
torch torchvision fastapi uvicorn matplotlib python-multipart streamlit<br>
```pip install torch torchvision fastapi uvicorn matplotlib python-multipart streamlit```<br>

# For Training (optional): <br>
- The training code is provided in the image_classification.ipynb <br>
- On running this file the dataset will be installed in the image_classification folder, also the trained model will be saved in a new directory named models in the image_classification folder as well.<br>
- In order to use the trained_model.pt, please edit the model_path variable in inference.py mentioned here <br>
_model_path = os.path.join( main_folder, 'best_model.pt' )_

After training the directory structure should look like this<br>
```
Directory Structure:-
main_folder
	|
	| -------- image_classification
	|		|
	| 		| —------- app.py
	|		|
	| 		| —------- best_model.pt
	| 		|
	| 		| —------- frontend.py
	|		|
	| 		| —------- image_classification.ipynb
	|		|
	| 		| —------- Image_Classification_Report.docx
	|		|
	| 		|—------- inference.py
	|		|
	| 		| —------- README.txt
	|		|
	| 		| —------- cifar10
	|		|	     |
	|		| 	     | —------ train
	|		| 	     |
	|		| 	     | —------ test
	|		|		
	| 		| —------- models
	|		|	    |
	|		| 	    | —------- trained_model.pt
```
# For inference:
Run the FastAPI from your terminal<br>
```uvicorn app:app --reload```<br>

- Open another instance of command prompt (windows) and terminal (linux and MacOS)
- enter the image_classification folder by changing directory appropriately.<br>
- Now activate the virtual environment using <br>
```image_classification\Scripts\activate``` (Windows)<br>
```source image_classification/bin/activate``` (MacOS, linux)<br>

Run the streamlit frontend on your localhost using the command<br>
```streamlit run frontend.py```<br>

The streamlit frontend will open in the browser, please upload either a ‘png’, ‘jpg’ or a ‘jpeg’ file by clicking on the Browse files button.<br>
