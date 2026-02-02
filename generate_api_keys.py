"""Generate API keys for FaceNet + MTCNN model access."""
import secrets
import json
from datetime import datetime

def generate_api_key() -> str:
    """Generate a secure API key."""
    return secrets.token_urlsafe(32)

def main():
    print("\n" + "="*70)
    print("🔑 API KEY GENERATOR - Face Recognition API")
    print("="*70)
    
    # Ask how many keys to generate
    try:
        num_keys = int(input("\n📊 How many API keys do you want to generate? (1-10): ") or "1")
        if num_keys < 1 or num_keys > 10:
            num_keys = 1
    except ValueError:
        num_keys = 1
    
    keys = []
    for i in range(num_keys):
        key = generate_api_key()
        keys.append({
            "key": key,
            "generated_at": datetime.now().isoformat(),
            "customer_id": f"customer_{i+1}",
            "status": "active"
        })
    
    # Display keys
    print(f"\n✅ Generated {num_keys} API key(s):\n")
    for i, key_info in enumerate(keys, 1):
        print(f"   {i}. {key_info['key']}")
        print(f"      Customer ID: {key_info['customer_id']}")
        print(f"      Generated: {key_info['generated_at']}")
        print()
    
    # Save to file
    with open("api_keys.json", "w") as f:
        json.dump(keys, f, indent=2)
    
    print("="*70)
    print("📝 Keys saved to: api_keys.json")
    print("="*70)
    
    print("\n📋 NEXT STEPS FOR MODAL DEPLOYMENT:")
    print("\n1️⃣  Choose ONE API key from above (or all for master key)")
    print("   This key will be used to protect your FaceNet/MTCNN endpoints")
    
    print("\n2️⃣  Create Modal Secret:")
    print("   modal secret create face-attendance-api-secrets \\")
    print(f"     FACE_API_KEY=<chosen_key_from_above> \\")
    print("     MONGODB_CONNECTION_STRING=<your_mongodb_uri>")
    
    print("\n3️⃣  Deploy to Modal:")
    print("   modal deploy get_started.py")
    
    print("\n4️⃣  Get your Modal endpoint URL:")
    print("   Visit: https://modal.com/apps")
    print("   Find: face-attendance-api")
    
    print("\n5️⃣  Test the API:")
    print("   curl -X POST https://your-modal-url/api/v1/detect-face \\")
    print("        -H 'x-api-key: YOUR_API_KEY' \\")
    print("        -H 'Content-Type: application/json' \\")
    print("        -d '{\"image\": \"base64_encoded_image\"}'")
    
    print("\n🔒 PROTECTED ENDPOINTS (require x-api-key header):")
    print("   - POST /api/v1/detect-face - Detect and identify faces")
    print("   - POST /api/v1/register-face - Register new face")
    print("   - POST /api/v1/verify-face - Verify face identity")
    
    print("\n✅ PUBLIC ENDPOINTS (no key needed):")
    print("   - GET  /health - Health check")
    print("   - POST /api/v1/auth/login - User login")
    print("   - POST /api/v1/auth/signup - User signup")
    
    print("\n💰 SELLING API KEYS:")
    print("   - Each key in api_keys.json can be sold to different customers")
    print("   - Track usage by customer_id")
    print("   - Implement rate limiting per key (not included in this basic version)")
    print("   - Consider using a database to track key usage and billing")
    
    print("\n" + "="*70)

if __name__ == "__main__":
    main()
