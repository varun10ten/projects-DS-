import requests
import json
import base64
import os

API_URL = "http://127.0.0.1:5000/generate-report"
OUTPUT_FILENAME = "financial_dashboard.png"
INPUT_JSON_FILE = "test_request.json"

def test_report_endpoint():
    print(f"--- Sending POST request to {API_URL} ---")

    # --- Load input JSON (MANDATORY step for POST API) ---
    request_payload = {}
    if os.path.exists(INPUT_JSON_FILE):
        print(f"Loading data from {INPUT_JSON_FILE}...")
        try:
            with open(INPUT_JSON_FILE, 'r') as f:
                request_payload = json.load(f)
        except json.JSONDecodeError:
            print(f"[ERROR] Could not decode JSON from {INPUT_JSON_FILE}. Skipping payload.")
            return # Exit if we can't load the required data
    else:
        print(f"[ERROR] Required input file '{INPUT_JSON_FILE}' not found. Cannot run test.")
        return # Exit if the file is missing
            
    try:
        # NOTE: We now use POST and send the request_payload as the JSON body
        response = requests.post(API_URL, json=request_payload) 
        
        response.raise_for_status() # Raise an exception for bad status codes (4xx or 5xx)
        
        # Parse the JSON response
        data = response.json()
        
        if data.get('status') == 'success':
            print("\n[SUCCESS] API returned status 'success'.")
            
            # --- 1. Print Text Results ---
            print("\n*** NLG Narrative Report ***")
            print(data.get('narrative_report', 'Narrative not found.'))
            
            print("\n*** Summary Table (Markdown) ***")
            print(data.get('summary_table_markdown', 'Table not found.'))
            
            # --- 2. Decode and Save Image ---
            base64_img = data.get('dashboard_image_base64')
            
            if base64_img:
                print(f"\n[IMAGE] Found image data ({len(base64_img):,} characters).")
                
                # Decode the Base64 string
                image_bytes = base64.b64decode(base64_img)
                
                # Save the decoded image bytes to a PNG file
                with open(OUTPUT_FILENAME, "wb") as f:
                    f.write(image_bytes)
                
                print(f"[SUCCESS] Dashboard image saved as: {OUTPUT_FILENAME}")
                print("Open the image file to verify the plot.")
            else:
                print("[WARNING] No Base64 image data found.")
                
        else:
            print(f"\n[FAILURE] API returned status 'error'. Message: {data.get('message', 'Unknown Error')}")
            
    except requests.exceptions.RequestException as e:
        print(f"\n[ERROR] Could not connect to the API. Is 'app.py' running? Error: {e}")
    except json.JSONDecodeError:
        print("\n[ERROR] Failed to decode JSON response.")
    except Exception as e:
        print(f"\n[CRITICAL ERROR] {e}")

if __name__ == "__main__":
    # Ensure the 'requests' library is installed: pip install requests
    test_report_endpoint()
