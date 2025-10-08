import requests
import json
import time

def test_ollama_connection():
    """Test if Ollama is running and responsive."""
    
    base_url = "http://localhost:11434"
    
    print("=" * 60)
    print("Testing Ollama Connection")
    print("=" * 60)
    
    # Test 1: Check if Ollama is running
    print("\n1. Checking if Ollama is running...")
    try:
        response = requests.get(f"{base_url}/api/tags", timeout=5)
        if response.status_code == 200:
            print("✅ Ollama is running!")
            models = response.json().get('models', [])
            print(f"   Available models: {len(models)}")
            for model in models:
                print(f"   - {model['name']}")
        else:
            print(f"❌ Ollama returned status code: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to Ollama!")
        print("   Make sure Ollama is running:")
        print("   - Windows: Check if Ollama app is running in system tray")
        print("   - Or run: ollama serve")
        return False
    except requests.exceptions.Timeout:
        print("❌ Ollama is not responding (timeout)")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False
    
    # Test 2: Check if llama3 is available
    print("\n2. Checking if llama3 model is available...")
    try:
        response = requests.get(f"{base_url}/api/tags", timeout=5)
        models = response.json().get('models', [])
        model_names = [m['name'] for m in models]
        
        if any('llama3' in name.lower() for name in model_names):
            print("✅ llama3 model found!")
        else:
            print("❌ llama3 model not found!")
            print("   Available models:", model_names)
            print("\n   To install llama3, run:")
            print("   ollama pull llama3")
            return False
    except Exception as e:
        print(f"❌ Error checking models: {e}")
        return False
    
    # Test 3: Try a simple generation
    print("\n3. Testing simple generation...")
    try:
        start_time = time.time()
        
        payload = {
            "model": "llama3",
            "prompt": "Say 'Hello, I am working!' and nothing else.",
            "stream": False,
            "options": {
                "temperature": 0,
                "num_predict": 50
            }
        }
        
        print("   Sending test prompt...")
        response = requests.post(
            f"{base_url}/api/generate",
            json=payload,
            timeout=30
        )
        
        elapsed = time.time() - start_time
        
        if response.status_code == 200:
            result = response.json()
            generated_text = result.get('response', '')
            print(f"✅ Generation successful! ({elapsed:.2f}s)")
            print(f"   Response: {generated_text[:100]}")
            
            if elapsed > 10:
                print("⚠️  WARNING: Generation is slow!")
                print("   This might cause timeouts in the app")
        else:
            print(f"❌ Generation failed with status: {response.status_code}")
            print(f"   Response: {response.text[:200]}")
            return False
            
    except requests.exceptions.Timeout:
        print("❌ Generation timed out after 30 seconds!")
        print("   Your system might be too slow for llama3")
        print("   Try a smaller model: ollama pull llama3.2:1b")
        return False
    except Exception as e:
        print(f"❌ Error during generation: {e}")
        return False
    
    print("\n" + "=" * 60)
    print("✅ All tests passed! Ollama is working correctly")
    print("=" * 60)
    return True

if __name__ == "__main__":
    test_ollama_connection()