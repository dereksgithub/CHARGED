# 🔧 Fix for Visualization Notebook KeyError

## The Problem

```python
KeyError: 'site'
```

This happens because when loading `sites.csv` with `index_col=0`, the 'site' column becomes the **index** (row labels), not a regular column!

## The Fix

### Option 1: Use the index from `iterrows()` ✅ RECOMMENDED

**Find this pattern in your notebook:**
```python
for _, site in sites.iterrows():
    popup = f"Site {site['site']}..."  # ❌ FAILS
```

**Replace with:**
```python
for site_id, site in sites.iterrows():
    popup = f"Site {site_id}..."  # ✅ WORKS
```

**Changes needed in your `create_city_time_map` function:**

```python
# OLD (line ~26):
popup=f"Site {site['site']}<br>Chargers: {site['charger_num']}<br>Total Volume: {site['total_volume']:.0f}"

# NEW:
popup=f"Site {site_id}<br>Chargers: {site['charger_num']}<br>Total Volume: {site['total_volume']:.0f}"
```

Make sure you also change the `iterrows()` line to capture the index:

```python
# OLD:
for _, site in sites.iterrows():

# NEW:
for site_id, site in sites.iterrows():
```

### Option 2: Load CSV without making 'site' the index

**Alternative approach - change how you load the CSV:**

```python
# Instead of:
sites = pd.read_csv(f'{data_path}sites.csv', index_col=0)

# Use:
sites = pd.read_csv(f'{data_path}sites.csv')  # 'site' stays as column
```

Then `site['site']` will work!

## Quick Fix Steps

1. **In your notebook cell that defines `create_city_time_map()`:**

   Find line ~22:
   ```python
   for _, site in sites.iterrows():
   ```

   Change to:
   ```python
   for site_id, site in sites.iterrows():
   ```

2. **Find line ~26:**
   ```python
   popup=f"Site {site['site']}<br>...
   ```

   Change to:
   ```python
   popup=f"Site {site_id}<br>...
   ```

3. **Re-run the cell** to define the function

4. **Run your map creation cell again**

## All Locations to Fix

Search your notebook for `site['site']` and replace with `site_id`:

1. **In `create_city_time_map()` function** (around line 26)
2. **In any comparison map function** (around line 451)
3. **In any clustering visualization** (around line 1145)

## Example: Complete Fixed Function

```python
def create_city_time_map(city_code, sample_hours=24*7):
    import folium
    from folium.plugins import HeatMapWithTime, MarkerCluster

    # Load data
    data_path = f'data/{city_code}_remove_zero/'
    sites = pd.read_csv(f'{data_path}sites.csv', index_col=0)  # 'site' is INDEX
    volume = pd.read_csv(f'{data_path}volume.csv', index_col=0, parse_dates=True)

    sites['total_volume'] = volume.sum(axis=0).values

    # Create map
    m = folium.Map(
        location=[sites['latitude'].mean(), sites['longitude'].mean()],
        zoom_start=11
    )

    marker_cluster = MarkerCluster(name='Charging Sites').add_to(m)

    # ✅ FIXED: Use site_id from iterrows()
    for site_id, site in sites.iterrows():  # Changed from: for _, site
        folium.CircleMarker(
            location=[site['latitude'], site['longitude']],
            radius=5,
            # ✅ FIXED: Use site_id instead of site['site']
            popup=f"Site {site_id}<br>Chargers: {site['charger_num']}<br>Total Volume: {site['total_volume']:.0f}",
            color='blue',
            fill=True,
            fillColor='blue',
            fillOpacity=0.6
        ).add_to(marker_cluster)

    # ... rest of function

    return m
```

## Why This Happens

When you load a CSV with `index_col=0`:
```python
sites = pd.read_csv('sites.csv', index_col=0)
```

Pandas uses the first column as the **row index** (like row names in Excel).

**Before (in CSV file):**
```
site,longitude,latitude,...
0,114.14,22.54,...
1,114.11,22.54,...
```

**After loading (in DataFrame):**
```
       longitude  latitude  ...
site
0      114.14     22.54     ...
1      114.11     22.54     ...
```

Notice: `site` is now the index, not a column!

## Verify the Fix

After making changes, test with:

```python
# Check that sites has the right structure
print(sites.index.name)  # Should print: 'site'
print(sites.columns.tolist())  # Should NOT include 'site'

# Test the function on one city
test_map = create_city_time_map('SZH', sample_hours=24)
test_map.save('test_map.html')
print("✓ Map created successfully!")
```

## Need Help?

Run this diagnostic in your notebook:

```python
# Diagnostic cell
data_path = 'data/SZH_remove_zero/'
sites = pd.read_csv(f'{data_path}sites.csv', index_col=0)

print("Index name:", sites.index.name)
print("Columns:", sites.columns.tolist())
print("\nFirst few rows:")
print(sites.head())
print("\nTo access site ID, use: site_id in iterrows()")
print("Example:")
for site_id, site in sites.head(2).iterrows():
    print(f"  Site ID: {site_id}, Lat: {site['latitude']}, Lon: {site['longitude']}")
```

This will confirm the structure and show you how to access the site IDs correctly.
