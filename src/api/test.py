import requests
import time
import random

def generate_test_data():
    return {
        "city": random.choice(["New York", "London", "Tokyo", "Paris", "Beijing"]),
        "temperature": random.uniform(15, 30),
        "humidity": random.uniform(30, 80),
        "wind_speed": random.uniform(0, 20),
        "pressure": random.uniform(980, 1020),
        "clouds": random.randint(0, 100),
        "co": random.uniform(0, 10),
        "no2": random.uniform(0, 100),
        "o3": random.uniform(0, 100),
        "so2": random.uniform(0, 100),
        "pm2_5": random.uniform(0, 50),
        "pm10": random.uniform(0, 100)
    }

def main():
    while True:
        try:
            response = requests.post(
                "http://localhost:8000/predict",
                json=generate_test_data()
            )
            print(f"Status: {response.status_code}, Response: {response.json()}")
        except Exception as e:
            print(f"Error: {e}")
        
        time.sleep(1)  # Send request every second

if __name__ == "__main__":
    main()