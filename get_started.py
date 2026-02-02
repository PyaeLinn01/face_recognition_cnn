"""Modal deployment for Face Attendance API with FaceNet + MTCNN."""
import os
import secrets
import modal

APP_NAME = "face-attendance-api"
API_KEY_ENV = "FACE_API_KEY"

# Create Modal image with all dependencies and copy local files
image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("libgl1", "libglib2.0-0", "libsm6", "libxext6", "libxrender-dev")
    .pip_install(
        "flask",
        "flask-cors",
        "numpy",
        "opencv-python-headless",
        "tensorflow==2.13.0",
        "mtcnn",
        "pymongo",
    )
    .add_local_file("api_server.py", "/root/api_server.py")
    .add_local_file("fr_utils.py", "/root/fr_utils.py")
    .add_local_file("fr.py", "/root/fr.py")
    .add_local_file("inception_blocks_v2.py", "/root/inception_blocks_v2.py")
    .add_local_file("nn4.small2.v7.h5", "/root/nn4.small2.v7.h5")
    .add_local_dir("weights", "/root/weights")
)

app = modal.App(APP_NAME, image=image)


def generate_api_key() -> str:
    """Generate a secure API key for protecting FaceNet/MTCNN endpoints."""
    return secrets.token_urlsafe(32)


@app.function(
    secrets=[modal.Secret.from_name("face-attendance-api-secrets")],
    timeout=3600,
    memory=4096,    gpu="any",  # Enable GPU for faster model inference)
@modal.asgi_app()
def flask_app():
    """Deploy Flask API on Modal with FaceNet + MTCNN."""
    import sys
    sys.path.insert(0, "/root")
    
    from api_server import app as flask_app
    return flask_app


@app.function()
@modal.fastapi_endpoint()
def healthcheck():
    """Simple health check endpoint."""
    return {"status": "ok", "service": "face-attendance-api", "deployed_on": "modal"}


@app.local_entrypoint()
def main():
    """Generate API key and show deployment instructions."""
    api_key = os.getenv(API_KEY_ENV)
    if not api_key:
        api_key = generate_api_key()
        print("\n" + "="*70)
        print("🔑 GENERATED API KEY (save this securely!)")
        print("="*70)
        print(f"\n{api_key}\n")
        print("="*70)
        print("\n📋 DEPLOYMENT STEPS:")
        print("\n1️⃣  Create Modal Secret:")
        print(f"   modal secret create face-attendance-api-secrets \\")
        print(f"     FACE_API_KEY={api_key} \\")
        print(f"     MONGODB_CONNECTION_STRING=<your_mongodb_uri>")
        print("\n2️⃣  Deploy to Modal:")
        print("   modal deploy get_started.py")
        print("\n3️⃣  Use the API:")
        print("   All requests to protected endpoints must include:")
        print(f"   Header: x-api-key: {api_key}")
        print("\n🔒 Protected endpoints:")
        print("   - POST /api/v1/detect-face")
        print("   - POST /api/v1/register-face")
        print("   - POST /api/v1/verify-face")
        print("\n✅ Public endpoints (no key needed):")
        print("   - GET  /health")
        print("   - POST /api/v1/auth/login")
        print("   - POST /api/v1/auth/signup")
        print("\n" + "="*70)
    else:
        print(f"\n✅ Using existing API key from environment: {api_key[:10]}...")
        print("\nTo deploy: modal deploy get_started.py")
