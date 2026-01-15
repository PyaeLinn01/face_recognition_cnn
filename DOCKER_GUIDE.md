# Docker Guide for Face Recognition CNN

This guide will help you build and run your face recognition application using Docker.

## Prerequisites

1. **Install Docker**: 
   - For macOS: Download from [Docker Desktop](https://www.docker.com/products/docker-desktop)
   - For Linux: `sudo apt-get install docker.io` (Ubuntu/Debian) or use your distribution's package manager
   - Verify installation: `docker --version`

2. **Ensure all project files are present**:
   - `app.py` (main application)
   - `requirements.txt` (Python dependencies)
   - `inception_blocks_v2.py` and `fr_utils.py` (model files)
   - `weights/` directory (contains model weights)
   - `images/` directory (contains reference face images)
   - `nn4.small2.v7.h5` (model weights file if needed)

## Building and Running with Docker Compose (Recommended - Includes MongoDB)

The easiest way to run the application with MongoDB is using Docker Compose, which runs MongoDB and Streamlit in separate containers.

1. **Navigate to your project directory**:
   ```bash
   cd /Users/pyaelinn/face_recon/face_recognition_cnn
   ```

2. **Start both MongoDB and Streamlit**:
   ```bash
   docker-compose up -d
   ```
   
   This will:
   - Pull the MongoDB 7.0 image
   - Build your Streamlit application image
   - Start both containers
   - Set up networking between them
   - Create persistent volumes for MongoDB data

3. **View logs**:
   ```bash
   docker-compose logs -f
   ```

4. **Stop everything**:
   ```bash
   docker-compose down
   ```

5. **Stop and remove volumes** (removes MongoDB data):
   ```bash
   docker-compose down -v
   ```

### MongoDB Connection Settings in Streamlit App

When using docker-compose, the MongoDB connection is automatically configured:
- **Connection String**: `mongodb://mongodb:27017/`
- **Database Name**: `face_attendance`

The app will automatically connect to MongoDB running in the separate container.

## Building the Docker Image (Without MongoDB)

If you want to build just the Streamlit app (and connect to an external MongoDB):

1. **Navigate to your project directory**:
   ```bash
   cd /Users/pyaelinn/face_recon/face_recognition_cnn
   ```

2. **Build the Docker image**:
   ```bash
   docker build -t face-recognition-cnn .
   ```
   
   This command:
   - `-t face-recognition-cnn`: Tags the image with the name "face-recognition-cnn"
   - `.`: Uses the current directory (where Dockerfile is located) as the build context
   
   **Note**: The first build may take several minutes as it downloads the base image and installs dependencies.

## Running the Container (Without Docker Compose)

### Basic Run

```bash
docker run -p 8501:8501 face-recognition-cnn
```

This command:
- `-p 8501:8501`: Maps port 8501 from the container to port 8501 on your host
- `face-recognition-cnn`: The image name to run

**Note**: If you need MongoDB, you'll need to either:
1. Use docker-compose (recommended - see above)
2. Run MongoDB separately: `docker run -d -p 27017:27017 --name mongodb mongo:7.0`
3. Connect to an external MongoDB instance

### Access the Application

Once the container is running, open your web browser and navigate to:
```
http://localhost:8501
```

You should see the Streamlit face attendance interface.

### Access MongoDB (if running separately)

If MongoDB is running in a container, you can access it:
```bash
docker exec -it mongodb mongosh
```

Or connect from your host machine:
```bash
mongosh mongodb://localhost:27017/
```

### Running in Detached Mode (Background)

To run the container in the background:

```bash
docker run -d -p 8501:8501 --name face-recognition face-recognition-cnn
```

- `-d`: Runs in detached mode (background)
- `--name face-recognition`: Gives the container a friendly name

### View Running Containers

```bash
docker ps
```

### Stop the Container

```bash
docker stop face-recognition
```

Or if you didn't name it:

```bash
docker stop <container-id>
```

### View Container Logs

```bash
docker logs face-recognition
```

Or to follow logs in real-time:

```bash
docker logs -f face-recognition
```

## Advanced Usage

### Run with Volume Mounting (for development)

If you want to modify code and see changes without rebuilding:

```bash
docker run -p 8501:8501 -v $(pwd):/app face-recognition-cnn
```

However, note that Streamlit may need to be restarted for code changes to take effect.

### Rebuild After Changes

If you modify the code or dependencies:

1. Stop the running container:
   ```bash
   docker stop face-recognition
   ```

2. Rebuild the image:
   ```bash
   docker build -t face-recognition-cnn .
   ```

3. Run again:
   ```bash
   docker run -d -p 8501:8501 --name face-recognition face-recognition-cnn
   ```

### Clean Up

**Remove a stopped container**:
```bash
docker rm face-recognition
```

**Remove the image**:
```bash
docker rmi face-recognition-cnn
```

**Remove all stopped containers**:
```bash
docker container prune
```

**Remove unused images**:
```bash
docker image prune
```

## Troubleshooting

### Port Already in Use

If port 8501 is already in use, use a different port:

```bash
docker run -p 8502:8501 face-recognition-cnn
```

Then access at `http://localhost:8502`

### Container Fails to Start

Check the logs:
```bash
docker logs face-recognition
```

### Permission Issues (Linux)

If you get permission denied errors, add your user to the docker group:

```bash
sudo usermod -aG docker $USER
```

Then log out and log back in.

### Out of Memory

If you encounter memory issues, increase Docker's memory limit in Docker Desktop settings (macOS/Windows) or ensure your system has enough RAM.

## Dockerfile Explanation

The Dockerfile:
1. Uses Python 3.10 slim base image
2. Installs system dependencies needed for OpenCV and MTCNN
3. Copies and installs Python dependencies from `requirements.txt`
4. Copies all project files
5. Exposes port 8501 (Streamlit's default port)
6. Runs the Streamlit app on startup

## Next Steps

- Consider using Docker Compose for more complex setups
- Set up environment variables for configuration
- Add health checks and monitoring
- Deploy to cloud platforms (AWS, GCP, Azure) using container services
