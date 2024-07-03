import numpy as np
import openmeteo_requests
import pandas as pd
from pysolar.solar import get_altitude, get_azimuth
import datetime
import matplotlib.pyplot as plt

# Define constants
STEFAN_BOLTZMANN_CONSTANT = 5.67e-8  
SPECIFIC_HEAT_CAPACITY_AIR = 1000  

window_orientations = {
    'south': 180,
    'north': 0,
    'east': 90,
    'west': 270
}

window_transmittance = 0.7


def calculate_aoi(datetime_obj, location, orientation):
    latitude, longitude = location["latitude"], location["longitude"]
    altitude = get_altitude(latitude, longitude, datetime_obj)
    azimuth = get_azimuth(latitude, longitude, datetime_obj)
    window_azimuth = window_orientations[orientation]
    aoi = abs(azimuth - window_azimuth)
    return aoi


class Building:
    def __init__(self, wall_area, wall_u_value, window_areas, window_u_value, roof_area, roof_u_value, thermal_mass,
                 air_volume):
        self.wall_area = wall_area
        self.wall_u_value = wall_u_value
        self.window_areas = window_areas
        self.window_u_value = window_u_value
        self.roof_area = roof_area
        self.roof_u_value = roof_u_value
        self.thermal_mass = thermal_mass
        self.air_volume = air_volume

    # Simulate heat transfer for a given time step
    def simulate_heat_transfer(self, outdoor_temp, indoor_temp, solar_radiation, time_step, datetime_obj, location):

        # Conduction heat transfer through walls
        q_wall = self.wall_area * self.wall_u_value * (outdoor_temp - indoor_temp) * time_step

        # Conduction heat transfer through roof
        q_roof = self.roof_area * self.roof_u_value * (outdoor_temp - indoor_temp) * time_step

        # Calculate total solar radiation gain through windows
        q_rad_windows = 0
        for orientation, area in self.window_areas.items():
            aoi = calculate_aoi(datetime_obj, location, orientation)
            if np.cos(np.radians(aoi)) > 0:  # Ensure positive contribution
                adjusted_solar_radiation = solar_radiation * np.cos(np.radians(aoi)) * window_transmittance
                q_rad_windows += area * adjusted_solar_radiation * time_step

        # Heat exchange with thermal mass (simplified model)
        q_thermal_mass = self.thermal_mass * (
                outdoor_temp - indoor_temp) * 0.1  # Adjust coefficient based on material properties

        # Total heat transfer
        # q_total = q_wall + q_rad_windows + q_roof + q_rad_windows - q_thermal_mass
        q_total = q_wall + q_roof + q_rad_windows - q_thermal_mass

        # Update indoor temperature (considering air exchange)
        air_mass = self.air_volume * 1.225  # Convert volume to mass assuming air density of 1.225 kg/m^3
        indoor_temp += (q_total - self.air_volume * natural_ventilation(self, indoor_temp, outdoor_temp, time_step)) / (
                air_mass * SPECIFIC_HEAT_CAPACITY_AIR)

        return indoor_temp


def natural_ventilation(self, indoor_temp, outdoor_temp, time_step):
    # Simulate air exchange rate based on wind speed and temperature difference
    wind_speed = 2  # m/s (adjust for specific conditions)
    stack_effect_pressure = abs(outdoor_temp - indoor_temp) * 0.034  # Pressure difference due to stack effect (Pa)
    effective_pressure = stack_effect_pressure + 0.5 * wind_speed ** 2 * 1.225  # Consider wind effect (Pa)
    orifice_coefficient = 0.6  # Adjust based on window opening area
    ach = effective_pressure * orifice_coefficient / (indoor_temp + 273.15) * 900 / time_step
    air_exchange_rate = ach * self.air_volume  # Volume of air exchanged per time step (m^3)
    return air_exchange_rate * (
            outdoor_temp - indoor_temp) * SPECIFIC_HEAT_CAPACITY_AIR  # Heat transfer due to air exchange (J)


def get_weather_data(location, weather_variables):
    """
  Fetches weather data from Open-Meteo API for a given location and desired variables.

  Args:
      location (dict): Dictionary containing latitude and longitude keys.
      weather_variables (list): List of weather variables to retrieve (e.g., ["temperature_2m", "precipitation"]).

  Returns:
      pandas.DataFrame: DataFrame containing weather data for the requested variables.
  """

    # Setup Open-Meteo API client (assuming cache and retry are already configured elsewhere)
    openmeteo = openmeteo_requests.Client()

    # Build the API request URL with location and weather variables
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": location["latitude"],
        "longitude": location["longitude"],
        "hourly": weather_variables
    }

    # Send the request and retrieve the response
    response = openmeteo.weather_api(url, params=params)[0]


    # Process hourly data
    hourly = response.Hourly()
    data = {}

    # Extract data for each requested variable
    for i, variable in enumerate(weather_variables):
        data[variable] = hourly.Variables(i).ValuesAsNumpy()

    for variable in data:
        data[variable] = data[variable].astype(float)

    # Create timestamps based on hourly data
    timestamps = pd.date_range(
        start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
        end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
        freq=pd.Timedelta(seconds=hourly.Interval()),
        inclusive="left"
    )

    # Combine data and timestamps into a DataFrame
    weather_data = pd.DataFrame(data=data, index=timestamps)

    return weather_data

def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size) / window_size, mode='valid')


def main():
    # Define building parameters
    building = Building(
        wall_area=100, wall_u_value=0.5,
        window_areas={'south': 10, 'east': 5, 'west': 5, 'north': 0},
        window_u_value=2.0,
        roof_area=50, roof_u_value=0.3,
        thermal_mass=10000, air_volume=50
    )

    # Get weather data for a specific location and time period
    location = {"latitude": 6.5244, "longitude": 3.3792} 
    weather_data = get_weather_data(location,
                                    weather_variables=["temperature_2m", "relative_humidity_2m", "wind_speed_10m",
                                                       "precipitation", "direct_normal_irradiance"])

    temperature_data = weather_data["temperature_2m"].values
    direct_normal_irridiance_data = weather_data["direct_normal_irradiance"].values
    timestamps = weather_data.index

    # Define simulation time step
    time_step = 1800  # Seconds

    # Initialize indoor temperature
    indoor_temp = 27  # Celsius 

    # Simulation loop
    simulation_results = []
    times = []
    outdoor_temps = []
    indoor_temps = []
    for i, (temperature, direct_normal_irridiance_data) in enumerate(
            zip(temperature_data, direct_normal_irridiance_data)):
        # Ensure temperature and solar_radiation are floats
        temperature = float(temperature)
        solar_radiation = float(direct_normal_irridiance_data)
        datetime_obj = timestamps[i].to_pydatetime()

        # Simulate heat transfer for the time step
        indoor_temp = building.simulate_heat_transfer(temperature, indoor_temp, solar_radiation, time_step,
                                                      datetime_obj, location)

        # Store simulation results (time, outdoor temp, indoor temp)
        simulation_results.append((i * time_step / 3600, temperature, indoor_temp))
   
    # Extract data for plotting
    times, outdoor_temps, indoor_temps = zip(*simulation_results)

    # Apply moving average to smooth indoor temperatures
    window_size = 5  # Adjust window size as needed
    smoothed_indoor_temps = moving_average(indoor_temps, window_size)

    # Adjust times for the moving average
    adjusted_times = times[:len(smoothed_indoor_temps)]

    # Plot the results
    plt.figure(figsize=(12, 6))
    plt.plot(times, outdoor_temps, label='Outdoor Temperature')
    plt.plot(adjusted_times, 
             smoothed_indoor_temps, 
             label='Indoor Temperature (Smoothed)')
    plt.xlabel('Time (hours)')
    plt.ylabel('Temperature (°C)')
    plt.title('Indoor and Outdoor Temperatures Over Time')
    plt.legend()
    plt.grid(True)
    plt.show()

# Run the simulation
if __name__ == "__main__":
    main()
