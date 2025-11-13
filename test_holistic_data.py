#!/usr/bin/env python3
"""
Test script to add some fake holistic requirements to test the detection logic
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from utils.postgres_vectorstore import PostgresVectorStore
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def add_test_holistic_requirements():
    """Add some test holistic requirements to verify the detection logic"""
    try:
        vs = PostgresVectorStore()
        filename = "URS Coating Machine Rev 1 - GLATT comments 03092025.docx"
        
        print("🧪 Adding test holistic requirements...")
        
        # Add a few test requirements with the holistic format
        test_requirements = [
            {
                "text": "The system shall provide automated coating process control with real-time monitoring capabilities.",
                "id": "HOLO_REQ_001"
            },
            {
                "text": "Temperature control shall maintain ±2°C accuracy throughout the coating process.",
                "id": "HOLO_REQ_002"
            },
            {
                "text": "The user interface shall display process parameters in real-time with 1-second refresh rate.",
                "id": "HOLO_REQ_003"
            }
        ]
        
        for i, req in enumerate(test_requirements):
            vs.add_document(
                text=req['text'],
                metadata={
                    "source_file": filename,
                    "page_number": 1,
                    "requirement_id": req['id'],  # This has HOLO_ prefix
                    "comments": ["source_type:gemini-holistic", "confidence:0.9", "category:functional"],
                    "responses": []
                }
            )
            print(f"  ✅ Added test requirement {i+1}: {req['id']}")
        
        vs.save()
        print(f"✅ Added {len(test_requirements)} test holistic requirements!")
        
        # Verify they were added
        print("\n🔍 Verifying added requirements...")
        reqs = vs.get_requirements_by_filename(filename)
        print(f"📊 Found {len(reqs)} total requirements for {filename}")
        
        holistic_count = 0
        for req in reqs:
            if req.get('requirement_id', '').startswith('HOLO_'):
                holistic_count += 1
                print(f"  🧠 Holistic: {req.get('requirement_id')} - {req.get('text', '')[:50]}...")
        
        print(f"✅ Found {holistic_count} holistic requirements that should be detected by Streamlit!")
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    add_test_holistic_requirements()