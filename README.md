# SalaryPrediction Project -Nano Grup

This is a basic Flask application Dockerized for easy deployment.

## Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/Nanogrup/SalaryPrediction.git
cd SalaryPrediction/flask_app
```
## Without Docker :
### 2. Install python 

### 3. Install dependencies(flask,...)
* You can download all the necessary libraries from the requirements.txt file with the command below:
```bash
pip install -r requirements.txt
```
* Or you can download each library individually with pip install as below:
```bash
pip install Flask
pip install ...
```

### 4. Start the flask app:
```bash
python app.py
```

### 5. Open a web browser and go to:
```
http://127.0.0.1:5000/
```

## With Docker : 
### 2. Install Docker

* Download and install Docker Desktop from the [Docker website](https://www.docker.com/products/docker-desktop).


### 3. Build the Docker image:
```bash
docker build -t flask-app .
```

### 4. Run the Docker container:
```bash
docker run -p 5000:5000 flask-app
```

### 5. Open a web browser and go to:
```
http://127.0.0.1:5000/
```

## With Virtual Environment(venv): 
### 2. Install Python
### 3. Create venv directory
```bash
python3 -m virtual environmentpath/to/venv
```
### 4. Activate virtual environment
```bash
source path/to/venv/bin/activate
```
### 5. Install dependencies(flask,...)
### 6. Start the flask app
### 7. Open a web browser and go to url

## Files and Directories

- `app.py`: Main Flask application file.
- `requirements.txt`: List of required Python packages.
- `Dockerfile`: Docker configuration file.
- `templates/`: Directory containing HTML templates.
- `static/`: Directory containing static files like CSS.
- `data/`: Directory containing csv file and jupyter notebooks.
- `machine_learning.py`: Model training application file.
- `salary_model.pkl`: Model file.
```csharp
flask_app/
    ├── app.py
    ├── machine_learning.py
    ├── requirements.txt
    ├── Dockerfile
    ├── salary_model.pkl
    ├── templates/
    │   ├── base.html
    │   ├── index.html
    │   ├── salary_prediction.html
    │   └── data_analysis.html
    │   └── result.html
    └── static/
        └── style.css
    ├── data/
        ├── Salary Analysis.ipynb
        ├── salaryData.csv
```