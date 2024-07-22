import requests

url = 'http://192.168.1.2:5000/predict'  # Update the URL if your server is running on a different address
file_path = '0acdc2b2-0dde-4073-8542-6fca275ab974___RS_LB 4857.JPG'  # Update this to the path of your image file

with open(file_path, 'rb') as file:
    files = {'file': file}
    response = requests.post(url, files=files)

print(response.json())