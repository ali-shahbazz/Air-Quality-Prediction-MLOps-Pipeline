import requests
import time
import random

def make_prediction(city, base_temp, base_humidity):
    """Make a prediction for a specific city with some randomization"""
    data = {
        "city": city,
        "temperature": base_temp + random.uniform(-2, 2),
        "humidity": base_humidity + random.uniform(-5, 5),
        "wind_speed": random.uniform(2, 15),
        "pressure": random.uniform(1000, 1020),
        "clouds": random.randint(0, 100),
        "co": random.uniform(0.4, 1.2),
        "no2": random.uniform(0.02, 0.08),
        "o3": random.uniform(0.03, 0.09),
        "so2": random.uniform(0.01, 0.05),
        "pm2_5": random.uniform(10, 35),
        "pm10": random.uniform(20, 45)
    }
    
    try:
        response = requests.post('http://localhost:8000/predict', json=data)
        if response.status_code == 200:
            print(f"Prediction for {city}: {response.json()['predicted_aqi']:.2f}")
        else:
            print(f"Error making prediction for {city}: {response.status_code}")
    except Exception as e:
        print(f"Exception making prediction for {city}: {str(e)}")

def main():
    # Cities with different baseline conditions
    cities = {
        "Beijing": (25, 60),    # Warm, moderate humidity
        "London": (18, 75),     # Mild, humid
        "New York": (22, 65),   # Moderate temp and humidity
        "Tokyo": (23, 70),      # Warm, humid
        "Paris": (20, 68),      # Mild, moderate humidity
        "Dubai": (35, 45),      # Hot, dry
        "Mumbai": (30, 80),     # Hot, very humid
        "Sydney": (24, 65)      # Warm, moderate humidity
    }
    
    print("Starting predictions...")
    
    # Make predictions every 30 seconds
    while True:
        for city, (base_temp, base_humidity) in cities.items():
            make_prediction(city, base_temp, base_humidity)
        
        print("\nWaiting 30 seconds before next round...\n")
        time.sleep(30)

if __name__ == "__main__":
    main()