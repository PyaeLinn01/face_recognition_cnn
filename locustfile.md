pip install locust

cd /Users/pyaelinn/face_recon/face_recognition_cnn
streamlit run attend_app.py --server.port 8501

cd /Users/pyaelinn/face_recon/face_recognition_cnn
locust -f locustfile.py --host=http://localhost:8501

Then open http://localhost:8089 in your browser. Set:

Number of users: 10 (start small)
Spawn rate: 2 (users per second)
Click "Start swarming"
