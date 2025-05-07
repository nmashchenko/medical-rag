#  MedChat: An AI-Powered Chat Interface for Medical Research Papers
You can run the web app with the provided bash script:  
```./start.sh```  
  
Here are instructions to run the web app manually:
### Frontend
Make sure you have npm and node.js installed.
Inside ```/frontend/``` run
- ```npm install```
- ```npm run dev```

This will run the frontend on ```localhost:3000```.

### Backend
Make sure you have Python installed.
Inside ```/backend/``` create a [virtual environment](https://docs.python.org/3/library/venv.html)
Then, inside the virtual environment, run
- ```pip install -r requirements.txt```
- ```python -m flask run```

This will run the backend on ```localhost:5000```.
If 'pdf_chunks.json' or 'embeddings.npy' doesn't exist, run ```python main.py``` to generate these files.

### Run via shell script
```chmod +x start.sh```
```./start.sh```
