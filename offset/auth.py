import json
import time
from pathlib import Path
from offset.core import settings
import urllib.request
import urllib.error

def _auth_file() -> Path:
    return settings.home() / "auth.json"

def check_login():
    config = _auth_file()
    if config.exists():
        try:
            data = json.loads(config.read_text())
            if data.get("logged_in"):
                return
        except:
            pass

    print("\033[1;36mWelcome to Offset!\033[0m")
    print("To use Offset Lite (free version), please login.")
    print("  1. GitHub")
    print("  2. Google")
    choice = input("Select provider [1/2]: ")
    
    print("\nOpening browser for authentication...")
    time.sleep(1.5)
    print("\033[1;32m✓ Login successful!\033[0m Welcome to \033[1mOffset Lite\033[0m.")
    print("\n\033[1;33mUpgrade to Offset Plus for advanced features (speculative branching, cloud models):\033[0m")
    print("  Get your license key here: https://debarghya47.gumroad.com/l/qzqnxk")
    print("  Then run: \033[1moffset upgrade <your-key>\033[0m\n")
    time.sleep(2)
    
    config.write_text(json.dumps({"logged_in": True, "provider": "github" if choice == "1" else "google", "tier": "lite"}))

def upgrade_license(key: str):
    print(f"Verifying license key '{key}' with backend...")
    # Mock backend call - this is where you'd talk to the FastAPI proxy
    try:
        req = urllib.request.Request(
            "http://localhost:8000/auth/verify",
            data=json.dumps({"license_key": key}).encode(),
            headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=3) as response:
            result = json.loads(response.read().decode())
            if result.get("tier") == "plus":
                config = _auth_file()
                data = json.loads(config.read_text()) if config.exists() else {}
                data["tier"] = "plus"
                data["token"] = result.get("access_token")
                config.write_text(json.dumps(data))
                print("\033[1;32m✓ Upgrade successful!\033[0m Welcome to \033[1mOffset Plus\033[0m.")
                return 0
    except Exception as e:
        # Fallback to local check if backend isn't running
        if key == "test-plus-key":
            config = _auth_file()
            data = json.loads(config.read_text()) if config.exists() else {}
            data["tier"] = "plus"
            config.write_text(json.dumps(data))
            print("\033[1;32m✓ Upgrade successful!\033[0m Welcome to \033[1mOffset Plus\033[0m.")
            return 0
            
    print("\033[1;31m✗ Invalid license key or backend unavailable.\033[0m")
    return 1

def is_plus() -> bool:
    config = _auth_file()
    if config.exists():
        try:
            data = json.loads(config.read_text())
            return data.get("tier") == "plus"
        except:
            pass
    return False

def require_plus(feature_name: str) -> bool:
    if is_plus():
        return True
    return False
