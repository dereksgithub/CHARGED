"""
Quick fix for the visualization notebook.
This script shows the corrected function to replace in your notebook.
"""

# The problem: When loading sites.csv, the 'site' column becomes the index
# Solution: Use site.name to get the index value instead of site['site']

# CORRECTED FUNCTION - Replace in your notebook:

def create_city_time_map(city_code, sample_hours=24*7):
    """
    Create an interactive time-series map for a city.
    FIXED VERSION - uses site.name instead of site['site']
    """
    import folium
    from folium.plugins import HeatMapWithTime, MarkerCluster
    import pandas as pd
    import numpy as np

    # Load data
    data_path = f'data/{city_code}_remove_zero/'
    sites = pd.read_csv(f'{data_path}sites.csv', index_col=0)  # site is index!
    volume = pd.read_csv(f'{data_path}volume.csv', index_col=0, parse_dates=True)

    # Calculate total volume for each site
    sites['total_volume'] = volume.sum(axis=0).values

    # Create base map
    center_lat = sites['latitude'].mean()
    center_lon = sites['longitude'].mean()

    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=11,
        tiles='OpenStreetMap'
    )

    # Add site markers with CORRECTED popup
    marker_cluster = MarkerCluster(name='Charging Sites').add_to(m)

    for site_id, site in sites.iterrows():  # site_id is the index value!
        folium.CircleMarker(
            location=[site['latitude'], site['longitude']],
            radius=5,
            # FIX: Use site_id (from iterrows) instead of site['site']
            popup=f"Site {site_id}<br>Chargers: {site['charger_num']}<br>Total Volume: {site['total_volume']:.0f}",
            color='blue',
            fill=True,
            fillColor='blue',
            fillOpacity=0.6
        ).add_to(marker_cluster)

    # Prepare heatmap data
    print(f"Preparing heatmap data...")

    # Sample time points
    time_index = volume.index[:sample_hours]

    # Create heatmap data: list of [time][site][lat, lon, weight]
    heat_data = []

    for time_point in time_index:
        time_data = []
        for site_id, site in sites.iterrows():
            if site_id in volume.columns:
                weight = volume.loc[time_point, site_id]
                if weight > 0:  # Only show sites with activity
                    time_data.append([
                        site['latitude'],
                        site['longitude'],
                        weight
                    ])
        heat_data.append(time_data)

    # Add heatmap
    HeatMapWithTime(
        heat_data,
        index=[t.strftime('%Y-%m-%d %H:%M') for t in time_index],
        auto_play=True,
        max_opacity=0.8,
        radius=15,
        blur=20,
        gradient={0.0: 'blue', 0.5: 'yellow', 1.0: 'red'},
        scale_radius=True,
        position='bottomleft'
    ).add_to(m)

    # Add layer control
    folium.LayerControl().add_to(m)

    return m


# ALTERNATIVE: If you want to keep 'site' as a column, load the CSV differently:

def load_sites_with_site_column(city_code):
    """
    Load sites.csv with 'site' as a regular column, not index
    """
    import pandas as pd

    data_path = f'data/{city_code}_remove_zero/'
    # Don't use index_col=0, so 'site' remains a column
    sites = pd.read_csv(f'{data_path}sites.csv')

    return sites


# QUICK FIX SUMMARY:
print("""
QUICK FIX FOR YOUR NOTEBOOK:
============================

The error occurs because when you use:
    sites = pd.read_csv('sites.csv', index_col=0)

The 'site' column becomes the INDEX, not a regular column!

FIX OPTION 1: Use the index value from iterrows()
-------------------------------------------------
for site_id, site in sites.iterrows():
    popup = f"Site {site_id}..."  # Use site_id, not site['site']

FIX OPTION 2: Don't use index_col=0
-----------------------------------
sites = pd.read_csv('sites.csv')  # Keep 'site' as column
for _, site in sites.iterrows():
    popup = f"Site {site['site']}..."  # Now site['site'] works!

RECOMMENDED: Use Option 1 (it's more efficient)

In your notebook, find all instances of:
    site['site']

And replace with:
    site_id  (when using: for site_id, site in sites.iterrows())
""")
