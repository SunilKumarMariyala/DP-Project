import requests
import sys
import os

def upload_file(file_path, url):
    """
    Upload a file to the specified URL
    """
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} does not exist")
        return
    
    print(f"Uploading file: {file_path} to {url}")
    
    with open(file_path, 'rb') as f:
        files = {'file': (os.path.basename(file_path), f, 'text/csv')}
        response = requests.post(url, files=files)
    
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.text}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python upload_file.py <file_path> [url]")
        sys.exit(1)
    
    file_path = sys.argv[1]
    url = sys.argv[2] if len(sys.argv) > 2 else "http://localhost:5000/api/matlab/process_realtime_data"
    
    upload_file(file_path, url)
