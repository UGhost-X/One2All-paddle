import base64
import json
import requests

with open('test/demo-4.jpg', 'rb') as f:
    img_b64 = base64.b64encode(f.read()).decode()

with open('test/img_b64.txt', 'w') as f:
    f.write(img_b64)

resp = requests.post('http://localhost:9979/predict_roi', json={'image': img_b64})
print(resp.json())