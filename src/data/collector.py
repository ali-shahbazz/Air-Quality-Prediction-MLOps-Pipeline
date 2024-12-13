# src/data/collector.py

import os
import json
import time
from datetime import datetime
import requests
from dotenv import load_dotenv
import pandas as pd
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class EnvironmentalDataCollector:
    def __init__(self):
        self.api_key = os.getenv('OPENWEATHER_API_KEY')
        if not self.api_key:
            raise ValueError("OpenWeather API key not found in environment variables")
            
        self.base_url = "http://api.openweathermap.org/data/2.5"
        self.cities = [
            {"name": "New York", "lat": 40.7128, "lon": -74.0060},
            {"name": "London", "lat": 51.5074, "lon": -0.1278},
            {"name": "Tokyo", "lat": 35.6762, "lon": 139.6503},
            {"name": "Paris", "lat": 48.8566, "lon": 2.3522},
            {"name": "Beijing", "lat": 39.9042, "lon": 116.4074},
            {"name": "Sydney", "lat": -33.8688, "lon": 151.2093},
            {"name": "Dubai", "lat": 25.2048, "lon": 55.2708},
            {"name": "Mumbai", "lat": 19.0760, "lon": 72.8777}
        ]
        
    def get_air_pollution(self, lat, lon):
        """Fetch air pollution data for given coordinates"""
        url = f"{self.base_url}/air_pollution?lat={lat}&lon={lon}&appid={self.api_key}"
        try:
            response = requests.get(url)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching air pollution data: {str(e)}")
            raise
    
    def get_weather(self, lat, lon):
        """Fetch weather data for given coordinates"""
        url = f"{self.base_url}/weather?lat={lat}&lon={lon}&appid={self.api_key}&units=metric"
        try:
            response = requests.get(url)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching weather data: {str(e)}")
            raise
    
    def collect_data(self):
        """Collect data for all cities and save to file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        data = []
        
        for city in self.cities:
            try:
                logger.info(f"Collecting data for {city['name']}")
                
                # Add delay to prevent API rate limiting
                time.sleep(1)
                
                weather = self.get_weather(city["lat"], city["lon"])
                pollution = self.get_air_pollution(city["lat"], city["lon"])
                
                record = {
                    "timestamp": timestamp,
                    "city": city["name"],
                    "temperature": weather["main"]["temp"],
                    "humidity": weather["main"]["humidity"],
                    "wind_speed": weather["wind"]["speed"],
                    "pressure": weather["main"]["pressure"],
                    "clouds": weather["clouds"]["all"],
                    "aqi": pollution["list"][0]["main"]["aqi"],
                    "co": pollution["list"][0]["components"]["co"],
                    "no2": pollution["list"][0]["components"]["no2"],
                    "o3": pollution["list"][0]["components"]["o3"],
                    "so2": pollution["list"][0]["components"]["so2"],
                    "pm2_5": pollution["list"][0]["components"]["pm2_5"],
                    "pm10": pollution["list"][0]["components"]["pm10"]
                }
                data.append(record)
                logger.info(f"Successfully collected data for {city['name']}")
                
            except Exception as e:
                logger.error(f"Error collecting data for {city['name']}: {str(e)}")
                continue
        
        if not data:
            raise ValueError("No data collected from any city")
        
        # Save to CSV
        df = pd.DataFrame(data)
        output_dir = Path("data/raw")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_file = output_dir / f"environmental_data_{timestamp}.csv"
        df.to_csv(output_file, index=False)
        logger.info(f"Data saved to {output_file}")
        
        # Save metadata
        metadata = {
            "timestamp": timestamp,
            "num_records": len(df),
            "cities_collected": [city["name"] for city in self.cities if city["name"] in df["city"].values],
            "cities_failed": [city["name"] for city in self.cities if city["name"] not in df["city"].values],
            "columns": list(df.columns)
        }
        
        metadata_file = output_dir / f"metadata_{timestamp}.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=4)
        
        return output_file

def main():
    try:
        collector = EnvironmentalDataCollector()
        output_file = collector.collect_data()
        logger.info("Data collection completed successfully")
    except Exception as e:
        logger.error(f"Data collection failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()