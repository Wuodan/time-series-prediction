import pandas as pd
import numpy as np
from pathlib import Path

def load_historical_data(file_paths):
	"""
	Load historical time series data from multiple files.
	Each file should contain a time-indexed DataFrame.
	"""
	historical_data = pd.concat([pd.read_csv(file_path, parse_dates=[0], index_col=0) for file_path in file_paths])
	historical_data = historical_data.sort_index()
	return historical_data

def find_nearest_comparison_days(target_date, historical_data, num_days=4):
	"""
	Find the nearest comparison days for the target date from the historical data.
	"""
	comparison_days = historical_data.loc[(historical_data.index.month == target_date.month) &
										  (historical_data.index.day == target_date.day)]

	# Sort by the absolute difference in years and take the closest `num_days`
	nearest_days = comparison_days.index.to_series().apply(lambda x: abs((x - target_date).days)).nsmallest(num_days).index

	return nearest_days

def apply_holiday_map(date, holiday_map):
	"""
	Apply holiday map to treat special days as a different weekday.
	"""
	if date in holiday_map:
		return holiday_map[date]
	return date.weekday()

def get_weekday_group(weekday, weekday_groups):
	"""
	Return the group label based on the supplied weekday groupings.
	"""
	for group, days in weekday_groups.items():
		if weekday in days:
			return group
	return None

def predict_next_year(historical_data, prediction_period, weekday_groups, holiday_map=None):
	"""
	Predict the next year's values based on the historical data and supplied parameters.
	"""
	predictions = []

	for target_date in pd.date_range(prediction_period[0], prediction_period[1], freq='60T'):

		# Adjust for holiday mapping if provided
		weekday = apply_holiday_map(target_date, holiday_map) if holiday_map else target_date.weekday()

		# Determine the group for the target date's weekday
		group = get_weekday_group(weekday, weekday_groups)

		# Find nearest comparison days based on calendar day
		comparison_days = find_nearest_comparison_days(target_date, historical_data)

		# Take the average of the corresponding 15-minute values from the comparison days
		comparison_data = historical_data.loc[comparison_days]
		avg_value = comparison_data[comparison_data.index.time == target_date.time()].mean()

		# Append the prediction
		predictions.append((target_date, avg_value))

	# Convert predictions to a DataFrame
	prediction_df = pd.DataFrame(predictions, columns=['Date', 'Predicted Value'])

	return prediction_df

# Example usage

# Get the absolute path to the 'samples/photos' folder relative to the repo root
repo_root = Path(__file__).parent.parent

# Load multiple historical data files
# file_paths = [repo_root / 'data/historical_data1.csv', repo_root / 'data/historical_data2.csv']  # Update with actual file paths
file_paths = [repo_root / 'data/historical_data1.csv']  # Update with actual file paths
historical_data = load_historical_data(file_paths)

# Define weekday groupings (e.g., Mon-Thu = [0, 1, 2, 3], Friday = [4], Saturday = [5], Sunday = [6])
weekday_groups = {
	'Mon-Thu': [0, 1, 2, 3],
	'Friday': [4],
	'Saturday': [5],
	'Sunday': [6]
}

# Holiday map setup (date -> weekday, e.g., 5 means treating as Saturday)
holiday_map = {pd.Timestamp('2024-12-25'): 5}  # Treat Christmas as Saturday

# Predict for next year (2025)
prediction_period = (pd.Timestamp('2025-01-01'), pd.Timestamp('2025-12-31'))

# Perform prediction
predicted_data = predict_next_year(historical_data, prediction_period, weekday_groups, holiday_map)

# Display predictions
# print(predicted_data.head())
print(predicted_data)
