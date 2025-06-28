# CHARGED Dataset Description

## Overview

The CHARGED (City-scale and Harmonized Dataset for Global Electric Vehicle Charging Demand Analysis) dataset provides comprehensive electric vehicle charging data and related auxiliary information for six major cities worldwide: **Los Angeles**, **São Paulo**, **Shenzhen**, **Amsterdam**, **Johannesburg**, and **Melbourne**.

## Dataset Versions

Each city has two data versions:
- **Standard version**: Complete dataset with all charging stations
- **`xxx_remove_zero` version**: Filtered dataset with charging stations that have zero charge volume, charge duration, and price removed for cleaner analysis

## Data Structure

Each city folder contains 10 CSV files. Due to size limitations, some files larger than 100MB are compressed in `csv-larger-than-100MB.rar`.

## File Descriptions

### Core Charging Data

#### volume.csv
- **Description**: Hourly charging volume (energy consumption) data
- **Format**: Time series with hourly granularity
- **Index**: Timestamps (hourly intervals)
- **Columns**: Station IDs
- **Unit**: Kilowatt-hours (kWh)
- **Use Case**: Primary target variable for demand prediction

#### duration.csv
- **Description**: Charging duration data representing time spent charging
- **Format**: Time series with hourly granularity
- **Index**: Timestamps (hourly intervals)
- **Columns**: Station IDs
- **Unit**: Hours (h)
- **Note**: Values represent charging duration between consecutive time intervals

### Pricing Information

#### e_price.csv
- **Description**: Electricity charging fees (energy cost)
- **Format**: Time series with hourly granularity
- **Index**: Timestamps (hourly intervals)
- **Columns**: Station IDs
- **Units by City**:
  - Los Angeles: USD/kWh
  - São Paulo: Brazilian Real (BRL)/kWh
  - Shenzhen: Chinese Yuan (CNY)/kWh
  - Amsterdam: Euro (EUR)/kWh
  - Johannesburg: South African Rand (ZAR)/kWh
  - Melbourne: Australian Dollar (AUD)/kWh

#### s_price.csv
- **Description**: Service fees (additional charges beyond electricity cost)
- **Format**: Time series with hourly granularity
- **Index**: Timestamps (hourly intervals)
- **Columns**: Station IDs
- **Note**: São Paulo, Johannesburg, and Melbourne have no service fees (uniformly set to 0)
- **Units**: Same as e_price.csv for each respective city

### Infrastructure Information

#### stations.csv
- **Description**: Comprehensive information about charging stations
- **Key Fields**:

| Field | Description | Unit |
|-------|-------------|------|
| station_id | Unique identifier for the charging station | N/A |
| longitude | Geographical longitude of station location | Degrees |
| latitude | Geographical latitude of station location | Degrees |
| pile_num | Number of charging piles at the station | Count |
| total_duration | Total charging duration recorded at station | Hours |
| total_volume | Total charging volume recorded at station | kWh |
| avg_power | Average charging power per charging record | kW |
| perimeter | Perimeter of DBSCAN cluster shape | Meters (m) |
| area | Area of DBSCAN cluster shape | Square meters (m²) |

#### piles.csv
- **Description**: Detailed information about individual charging piles
- **Key Fields**:

| Field | Description | Unit |
|-------|-------------|------|
| pile_id | Unique identifier for the charging pile | N/A |
| longitude | Geographical longitude of pile location | Degrees |
| latitude | Geographical latitude of pile location | Degrees |
| station_id | Identifier of the station containing this pile | N/A |
| total_duration | Total charging duration at this pile | Hours |
| total_volume | Total energy delivered at this pile | kWh |
| avg_power | Average charging power per charging record | kW |

### Auxiliary Data

#### distance.csv
- **Description**: Inter-station distance matrix
- **Format**: Symmetric matrix
- **Index/Columns**: Station IDs
- **Calculation**: Geodesic distance based on ellipsoidal Earth model
- **Unit**: Kilometers (km)
- **Use Case**: Spatial analysis and clustering

#### weather.csv
- **Description**: Hourly weather data from Visual Crossing API
- **Source**: [Visual Crossing Weather API](https://www.visualcrossing.com/resources/documentation/weather-api/timeline-weather-api/)
- **Key Fields**:

| Field | Description | Unit |
|-------|-------------|------|
| temp | Temperature | Celsius (°C) |
| feelslike | Apparent temperature (heat index/wind chill) | Celsius (°C) |
| humidity | Relative humidity | Percentage (%) |
| dew | Dew point temperature | Celsius (°C) |
| precip | Liquid precipitation amount | mm |
| snow | Snowfall amount | mm |
| snowdepth | Snow depth on ground | mm |
| preciptype | Precipitation type | Array (rain/snow/freezingrain/ice) |
| windgust | Instantaneous wind speed | km/h |
| windspeed | Sustained wind speed | km/h |
| winddir | Wind direction | Degrees |
| pressure | Sea level atmospheric pressure | hPa |
| visibility | Visibility distance | km |
| cloudcover | Cloud coverage | Percentage (0-100%) |
| solarradiation | Solar radiation power | W/m² |
| solarenergy | Solar energy accumulation | MJ/m² |
| uvindex | UV exposure index | Scale (0-10) |
| conditions | Weather conditions | Categorical (see mapping below) |

**Weather Conditions Mapping**:
- 0: Clear
- 1: Overcast
- 2: Partially cloudy
- 3: Rain
- 4: Rain, Fog
- 5: Rain, Overcast
- 6: Rain, Partially cloudy
- 7: Snow
- 8: Snow, Fog
- 9: Snow, Partially cloudy
- 10: Snow, Rain
- 11: Snow, Rain, Overcast
- 12: Snow, Rain, Partially cloudy

#### poi.csv
- **Description**: Points of Interest (POIs) from OpenStreetMap
- **Source**: [OpenStreetMap](https://www.openstreetmap.org/)
- **Categories**: Amenities, buildings, offices, shops, leisure facilities, etc.
- **Reference**: [OSM Wiki - Map Features](https://wiki.openstreetmap.org/wiki/Map_features)
- **Fields**:

| Field | Description | Unit |
|-------|-------------|------|
| type | Geographic feature category | N/A |
| longitude | POI longitude | Degrees |
| latitude | POI latitude | Degrees |

### Metadata

#### info.csv
- **Description**: Dataset metadata and summary statistics
- **Key Fields**:

| Field | Description | Unit |
|-------|-------------|------|
| city | City name | N/A |
| country | Country name | N/A |
| abbreviation | City code (3-letter) | N/A |
| total_piles | Total number of charging piles | Count |
| total_stations | Total number of charging stations | Count |
| DBSCAN_eps | DBSCAN clustering radius parameter | Meters (m) |
| total_duration | Total charging duration in dataset | Hours |
| total_volume | Total charging volume in dataset | kWh |
| avg_power | Average charging power across all records | kW |

## Data Quality Notes

### Preprocessing Applied
- **Anomaly Detection**: Outliers identified and corrected using IQR method
- **Zero Sequence Handling**: Continuous zero sequences interpolated
- **Station Clustering**: Geographic clustering using DBSCAN algorithm
- **Data Validation**: Cross-checked for consistency and completeness

### Known Limitations
- Some cities may have missing data during certain time periods
- Weather data availability depends on API coverage
- POI data completeness varies by city and OSM coverage
- Currency exchange rates not provided for cross-city comparisons

## Usage Guidelines

### Recommended Workflow
1. Start with `xxx_remove_zero` versions for cleaner analysis
2. Use `info.csv` to understand dataset characteristics
3. Combine charging data with weather and POI data for comprehensive analysis
4. Apply appropriate temporal and spatial aggregation as needed

### Data Loading Example
```python
import pandas as pd

# Load charging volume data
volume_data = pd.read_csv('data/SZH_remove_zero/volume.csv', index_col=0)
volume_data.index = pd.to_datetime(volume_data.index)

# Load station information
stations = pd.read_csv('data/SZH_remove_zero/stations.csv')

# Load weather data
weather = pd.read_csv('data/SZH_remove_zero/weather.csv', index_col=0)
weather.index = pd.to_datetime(weather.index)
```

## Citation

If you use this dataset in your research, please cite:

```bibtex
@article{charged2024,
  title={CHARGED: A City-scale and Harmonized Dataset for Global Electric Vehicle Charging Demand Analysis},
  author={Your Name},
  journal={arXiv preprint},
  year={2024}
}
```

## Support

For questions about the dataset, please refer to:
- Main project documentation: [README.md](../README.md)
- GitHub Issues: [Project Issues](https://github.com/IntelligentSystemsLab/CHARGED/issues)
- Contact: guozh29@mail2.sysu.edu.cn

