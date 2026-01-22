# Face Attendance System - Full Stack Setup Guide

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    React Frontend                            │
│        (face-attend-main/)                                   │
│     - Face Registration UI                                   │
│     - Face Verification UI                                   │
│     - Attendance Records Display                             │
└──────────────────────────┬──────────────────────────────────┘
                           │ HTTP/REST
                           ▼
┌─────────────────────────────────────────────────────────────┐
│              Flask REST API Backend                          │
│        (api_server.py)                                       │
│     - /register-face endpoint                                │
│     - /verify-face endpoint                                  │
│     - /attendance/record endpoint                            │
│     - /attendance/recent endpoint                            │
│     - /faces/list endpoint                                   │
└──────────────────┬──────────────────────────┬────────────────┘
                   │                          │
                   ▼                          ▼
        ┌──────────────────┐      ┌──────────────────┐
        │   ML Models      │      │   MongoDB        │
        │  - FaceNet       │      │  - Faces Collection│
        │  - MTCNN         │      │  - Attendance    │
        └──────────────────┘      └──────────────────┘
```

## Step 1: Backend Setup

### 1.1 Install Dependencies

```bash
cd /Users/pyaelinn/face_recon/face_recognition_cnn

# Install API dependencies
pip install -r requirements-api.txt

# Or install individually
pip install Flask==3.0.0 Flask-CORS==4.0.0 python-dotenv==1.0.0
```

### 1.2 Set Environment Variables

Create a `.env` file in the project root:

```bash
cat > .env << EOF
# MongoDB Configuration
MONGODB_CONNECTION_STRING=mongodb://localhost:27017/
MONGODB_DATABASE_NAME=face_attendance

# API Configuration
API_PORT=5000
FLASK_ENV=development
EOF
```

**For MongoDB Atlas (Cloud):**
```bash
MONGODB_CONNECTION_STRING=mongodb+srv://<username>:<password>@<cluster>.mongodb.net/?retryWrites=true&w=majority
MONGODB_DATABASE_NAME=face_attendance
```

### 1.3 Verify MongoDB Connection

```bash
python -c "from pymongo import MongoClient; client = MongoClient('mongodb://localhost:27017/'); print('✅ MongoDB connected' if client.server_info() else '❌ Failed')"
```

### 1.4 Start the Backend API Server

```bash
# Terminal 1: Start API server
python api_server.py

# Output should show:
# WARNING: This is a development server. Do not use it in production.
# Running on http://0.0.0.0:5000
```

### 1.5 Test API Health

```bash
curl http://localhost:5000/health

# Expected response:
# {
#   "status": "ok",
#   "service": "Face Attendance API",
#   "models": {"facenet": "loaded", "mtcnn": "available"},
#   "mongodb": "connected"
# }
```

---

## Step 2: Frontend Setup

### 2.1 Install Dependencies

```bash
cd /Users/pyaelinn/face_recon/face_recognition_cnn/face-attend-main

# Using npm
npm install

# Or using bun (faster)
bun install
```

### 2.2 Configure Environment

```bash
# Copy example environment
cp .env.example .env.local

# Edit .env.local to match your setup
cat .env.local
# VITE_API_URL=http://localhost:5000/api/v1
# VITE_ENV=development
```

### 2.3 Start Frontend Development Server

```bash
# Terminal 2: Start frontend
npm run dev
# or
bun run dev

# Output should show:
# Local:        http://localhost:5173
# press h to show help
```

---

## Step 3: Complete Workflow

### 3.1 Register a Face

1. Open http://localhost:5173
2. Navigate to "Register Face" tab
3. Enter name: "John Doe"
4. Click "Start Camera"
5. Capture 4 photos from different angles
6. Click "Register Face"
7. Backend stores in MongoDB

**MongoDB stores:**
```json
{
  "_id": ObjectId(...),
  "name": "John Doe",
  "image_index": 1,
  "image_bytes": BinData(...),
  "created_at": ISODate(...)
}
```

### 3.2 Verify & Record Attendance

1. Navigate to "Attendance" tab
2. Click "Start Camera"
3. Position your face in frame
4. System verifies against registered faces
5. Shows match result with confidence
6. Records attendance if matched

**Stored in MongoDB:**
```json
{
  "_id": ObjectId(...),
  "timestamp": ISODate(...),
  "entered_name": "John Doe",
  "matched_identity": "John Doe",
  "distance": 0.45
}
```

---

## Step 4: API Endpoints

### Register Face
```bash
POST /api/v1/register-face
Content-Type: application/json

{
  "name": "John Doe",
  "image_base64": "<base64_encoded_image>",
  "image_index": 1
}

Response (201):
{
  "success": true,
  "id": "507f1f77bcf86cd799439011",
  "name": "John Doe",
  "index": 1
}
```

### Verify Face
```bash
POST /api/v1/verify-face
Content-Type: application/json

{
  "image_base64": "<base64_encoded_image>",
  "threshold": 0.7,
  "use_detection": true
}

Response (200):
{
  "match": true,
  "identity": "John Doe",
  "distance": 0.45,
  "threshold": 0.7,
  "confidence": 0.55
}
```

### Record Attendance
```bash
POST /api/v1/attendance/record
Content-Type: application/json

{
  "name": "John Doe",
  "identity": "John Doe",
  "distance": 0.45,
  "timestamp": "2024-01-22T10:30:00"
}

Response (201):
{
  "success": true,
  "id": "507f1f77bcf86cd799439012",
  "recorded_at": "2024-01-22T10:30:00"
}
```

### Get Recent Attendance
```bash
GET /api/v1/attendance/recent?limit=50

Response (200):
{
  "count": 12,
  "records": [
    {
      "_id": "507f1f77bcf86cd799439013",
      "timestamp": "2024-01-22T10:30:00",
      "entered_name": "John Doe",
      "matched_identity": "John Doe",
      "distance": 0.45
    },
    ...
  ]
}
```

### List Registered Faces
```bash
GET /api/v1/faces/list

Response (200):
{
  "count": 3,
  "faces": [
    {"name": "Alice Smith", "images": 4},
    {"name": "Bob Johnson", "images": 4},
    {"name": "John Doe", "images": 4}
  ]
}
```

---

## Step 5: Deployment

### 5.1 Docker Deployment

**Backend Dockerfile:**
```dockerfile
FROM python:3.10-slim

WORKDIR /app
COPY requirements.txt requirements-api.txt ./
RUN pip install -r requirements.txt -r requirements-api.txt

COPY . .

EXPOSE 5000
CMD ["python", "api_server.py"]
```

**Frontend Dockerfile:**
```dockerfile
FROM node:18-alpine AS build
WORKDIR /app
COPY face-attend-main/package*.json ./
RUN npm install
COPY face-attend-main ./
RUN npm run build

FROM nginx:alpine
COPY --from=build /app/dist /usr/share/nginx/html
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
```

**docker-compose.yml:**
```yaml
version: '3.8'

services:
  mongodb:
    image: mongo:7
    ports:
      - "27017:27017"
    volumes:
      - mongodb_data:/data/db
    environment:
      MONGO_INITDB_DATABASE: face_attendance

  backend:
    build:
      context: .
      dockerfile: Dockerfile
    ports:
      - "5000:5000"
    environment:
      MONGODB_CONNECTION_STRING: mongodb://mongodb:27017/
      MONGODB_DATABASE_NAME: face_attendance
      API_PORT: 5000
    depends_on:
      - mongodb

  frontend:
    build:
      context: .
      dockerfile: Dockerfile.frontend
    ports:
      - "80:80"
    depends_on:
      - backend

volumes:
  mongodb_data:
```

**Run with Docker:**
```bash
docker-compose up -d

# Access at http://localhost
```

### 5.2 AWS Deployment

**For Backend (App Runner):**
```bash
# Build and push to ECR
aws ecr create-repository --repository-name face-attendance-api

docker build -t face-attendance-api:latest .
docker tag face-attendance-api:latest <account>.dkr.ecr.<region>.amazonaws.com/face-attendance-api:latest
docker push <account>.dkr.ecr.<region>.amazonaws.com/face-attendance-api:latest

# Create App Runner service pointing to this image
# Set environment variables:
# MONGODB_CONNECTION_STRING=<your_mongodb_url>
# MONGODB_DATABASE_NAME=face_attendance
```

**For Frontend (S3 + CloudFront or Amplify):**
```bash
cd face-attend-main
npm run build

# Upload to S3
aws s3 sync dist/ s3://your-bucket-name --delete

# Or use AWS Amplify:
# - Connect GitHub repo
# - Set build command: npm run build
# - Set output directory: dist
```

---

## Step 6: Troubleshooting

### MongoDB Connection Failed
```bash
# Check MongoDB is running
mongodb_pid=$(pgrep -f "mongod")
if [ -z "$mongodb_pid" ]; then echo "Not running"; else echo "Running: $mongodb_pid"; fi

# Or start with Docker
docker run -d -p 27017:27017 --name mongodb mongo:7
```

### API Port Already in Use
```bash
# Check what's using port 5000
lsof -i :5000

# Kill the process
kill -9 <PID>

# Or use different port
API_PORT=5001 python api_server.py
```

### CORS Errors
- Make sure `VITE_API_URL` matches your backend URL
- Backend already has CORS enabled with `flask-cors`
- Check browser console for exact error

### Camera Permission Denied
- Open Chrome settings → Privacy and security → Site settings
- Allow camera access for localhost

---

## Step 7: Load Testing

```bash
# Install Locust
pip install locust

# Run load test against frontend
locust -f locustfile.py --host=http://localhost:5173 --users 10 --spawn-rate 2

# Run load test against API
locust -f locustfile.py --host=http://localhost:5000 --users 10 --spawn-rate 2
```

---

## Production Checklist

- [ ] Use MongoDB Atlas (cloud) instead of local MongoDB
- [ ] Deploy Backend to AWS App Runner or Fargate
- [ ] Deploy Frontend to S3 + CloudFront or AWS Amplify
- [ ] Enable HTTPS/TLS certificates
- [ ] Set up monitoring (CloudWatch logs)
- [ ] Configure auto-scaling (App Runner)
- [ ] Set up backup strategy for MongoDB
- [ ] Enable authentication (JWT tokens)
- [ ] Run security scanning
- [ ] Load test with real traffic patterns
- [ ] Set up CI/CD pipeline (GitHub Actions)

---

## File Structure

```
face_recognition_cnn/
├── api_server.py              # Flask REST API backend
├── attend_app.py              # Streamlit app (legacy UI)
├── app.py                     # Core ML functions
├── fr_utils.py                # FaceNet utilities
├── inception_blocks_v2.py     # Model architecture
├── requirements.txt           # Streamlit deps
├── requirements-api.txt       # API deps
├── nn4.small2.v7.h5          # FaceNet weights
├── face-attend-main/          # React frontend
│   ├── src/
│   │   ├── lib/
│   │   │   └── face-api.ts    # API client
│   │   ├── pages/
│   │   │   ├── FaceRegister.tsx
│   │   │   └── MarkAttendance.tsx
│   │   └── App.tsx
│   ├── package.json
│   └── .env.example
└── images/                    # Registered faces (local fallback)
```

---

## Quick Start (Local Development)

```bash
# Terminal 1: Start backend
cd /Users/pyaelinn/face_recon/face_recognition_cnn
python api_server.py

# Terminal 2: Start frontend
cd /Users/pyaelinn/face_recon/face_recognition_cnn/face-attend-main
npm run dev

# Open browser to http://localhost:5173
```
