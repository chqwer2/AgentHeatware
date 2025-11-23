from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple
import csv
import math


@dataclass
class EnvironmentState:
    day: int
    temperature: float  # °C
    humidity: float     # 0..1
    ac_available: bool


def load_weather_from_csv(path: str) -> List[Tuple[float, float]]:
    """
    Load weather data as a list of (temperature, humidity) tuples.

    Expected CSV columns:
      date,temperature,humidity

    Humidity may be 0-1 or 0-100; if >1 we scale to 0-1.
    """
    print(f"[INIT] Loading weather from {path} ...")
    series: List[Tuple[float, float]] = []
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            temp = float(row["temperature"])
            hum = float(row["humidity"])
            if hum > 1.0:
                hum /= 100.0
            series.append((temp, hum))
    print(f"[INIT] Loaded {len(series)} weather days.")
    return series


def generate_dummy_weather(days: int = 14) -> List[Tuple[float, float]]:
    """
    Synthetic heatwave pattern if you don't have real data yet.
    """
    print(f"[INIT] Generating dummy weather for {days} days ...")
    series: List[Tuple[float, float]] = []
    for d in range(days):
        t = d / max(1, days - 1)
        heat_factor = math.sin(math.pi * t)
        temp = 28.0 + heat_factor * 15.0   # 28 -> 43
        hum = 0.4 + heat_factor * 0.3      # 0.4 -> 0.7
        series.append((temp, hum))
    print("[INIT] Dummy weather generated.")
    return series
