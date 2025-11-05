"""
Test script for Gemini Pro integration.
Run this to verify that Gemini is working correctly with your API key.
"""

import sys
import os
sys.path.append('.')

def test_gemini_basic():
    """Test basic Gemini functionality."""
    try:
        from utils.gemini_processor import GeminiProcessor, test_gemini_processor
        
        # Get API key
        api_key = input("Enter your Gemini API key: ").strip()
        if not api_key:
            print("No API key provided. Exiting.")
            return
        
        print("Testing Gemini Pro integration...")
        print("=" * 50)
        
        # Test with sample URS text
        sample_text = """
        6.1 CIP unit shall follow the 4 cleaning phases steps as follows:
        - Pre-rinse phase
        - Caustic wash phase  
        - Intermediate rinse phase
        - Sanitization phase

        6.2 All electrical wiring shall be concealed and with proper earthing arrangement.

        6.3 The equipment must comply with safety standards and include:
        • Emergency stop buttons accessible from all positions
        • Safety interlocks on all access doors
        • Proper grounding of all electrical components

        7.1 CIP return pump shall have the following specifications:
        Flow rate: 150 L/min minimum
        Head: 25 meters minimum
        Material: 316L stainless steel

        Process Parameters:
        - Temperature range: 60-80°C for caustic wash
        - pH levels: 12-14 for cleaning solution
        - Contact time: minimum 15 minutes per phase
        """
        
        # Run test
        requirements = test_gemini_processor(api_key, sample_text)
        
        print(f"\n✅ Test completed successfully!")
        print(f"🔍 Found {len(requirements)} requirements")
        
        # Show categorization
        categories = {}
        for req in requirements:
            cat = req.get('category', 'unknown')
            if cat not in categories:
                categories[cat] = 0
            categories[cat] += 1
        
        print(f"📊 Categories: {categories}")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_config():
    """Test configuration loading."""
    try:
        from config import GEMINI_API_KEY, USE_GEMINI_PREPROCESSING
        print(f"Config loaded: USE_GEMINI_PREPROCESSING = {USE_GEMINI_PREPROCESSING}")
        if GEMINI_API_KEY == "your_gemini_api_key_here":
            print("⚠️  Please update your API key in config.py")
        else:
            print("✅ API key configured in config.py")
        return True
    except ImportError as e:
        print(f"❌ Config import failed: {e}")
        return False

if __name__ == "__main__":
    print("🧪 Gemini Pro Integration Test")
    print("=" * 40)
    
    # Test configuration
    print("\n1. Testing configuration...")
    config_ok = test_config()
    
    # Test basic functionality
    print("\n2. Testing Gemini functionality...")
    if config_ok:
        test_gemini_basic()
    else:
        print("Skipping Gemini test due to config issues.")
    
    print("\nTest complete!")