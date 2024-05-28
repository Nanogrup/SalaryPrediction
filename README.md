# SalaryPrediction Project -Nano Grup

This is a basic Flask application Dockerized for easy deployment.

## Setup Instructions

### 1. Install Docker

#### For macOS:

1. **Docker Desktop**: Download and install Docker Desktop from the [Docker website](https://www.docker.com/products/docker-desktop).

2. **System Requirements**: Ensure your macOS version meets the [Docker Desktop system requirements](https://docs.docker.com/desktop/install/#what-to-know-before-you-install).

#### For Windows:

1. **Docker Desktop**: Download and install Docker Desktop from the [Docker website](https://www.docker.com/products/docker-desktop).

2. **System Requirements**: Ensure your Windows version meets the [Docker Desktop system requirements](https://docs.docker.com/desktop/install/#what-to-know-before-you-install).

#### For Linux:

1. **Ubuntu**: Follow the [official Docker installation instructions for Ubuntu](https://docs.docker.com/engine/install/ubuntu/).

2. **Debian**: Follow the [official Docker installation instructions for Debian](https://docs.docker.com/engine/install/debian/).

3. **CentOS**: Follow the [official Docker installation instructions for CentOS](https://docs.docker.com/engine/install/centos/).


### 2. Clone the Repository

```bash
git clone https://github.com/Nanogrup/SalaryPrediction.git
cd SalaryPrediction/flask_app
```

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



## Files and Directories

- `app.py`: Main Flask application file.
- `requirements.txt`: List of required Python packages.
- `Dockerfile`: Docker configuration file.
- `templates/`: Directory containing HTML templates.
- `static/`: Directory containing static files like CSS.

```csharp
flask_app/
    ├── app.py
    ├── requirements.txt
    ├── Dockerfile
    ├── templates/
    │   ├── base.html
    │   ├── index.html
    │   ├── salary_prediction.html
    │   └── data_analysis.html
    └── static/
        └── style.css
```