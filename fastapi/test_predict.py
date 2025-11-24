import requests

url = "http://127.0.0.1:8000/predict"

payload = {
    "rel_compact": 0.75,
    "surface_area": 650.0,
    "wall_area": 320.0,
    "roof_area": 200.0,
    "overall_height": 4.5,
    "orientation": 2,
    "glazing_area": 0.15,
    "glazing_dist": 5
}

response = requests.post(url, json=payload)

print("STATUS:", response.status_code)
print("TEXT:", response.text)
print("HEADERS:", response.headers)
