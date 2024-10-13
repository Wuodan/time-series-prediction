import pandas as pd
from pathlib import Path
import argparse
import os

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

def predict_next_year(file_paths, prediction_period, weekday_groups, holiday_map=None):
	"""
	Predict the next year's values based on the historical data and supplied parameters.

	Parameters:
	- file_paths: List of file paths containing historical time series data
	- prediction_period: Tuple of start and end date for the prediction (e.g., ('2025-01-01', '2025-12-31'))
	- weekday_groups: Dictionary specifying how weekdays are grouped (e.g., {'Mon-Thu': [0,1,2,3], 'Friday': [4], 'Saturday': [5], 'Sunday': [6]})
	- holiday_map: Dictionary of specific dates to be treated as another weekday (optional)

	Returns:
	- A DataFrame with predicted values for the specified period
	"""
	# Load multiple historical data files
	historical_data = load_historical_data(file_paths)

	predictions = []

	# Convert prediction period strings to Timestamps
	start_date, end_date = pd.Timestamp(prediction_period[0]), pd.Timestamp(prediction_period[1])

	for target_date in pd.date_range(start_date, end_date, freq='60T'):

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

def main():
	# Create an argument parser
	parser = argparse.ArgumentParser(description='Predict future values based on historical time series data.')

	# Add arguments
	parser.add_argument('--file_paths', nargs='+', required=True, help='Paths to historical data CSV files.')
	parser.add_argument('--start_date', required=True, help='Start date for the prediction period (e.g., 2025-01-01).')
	parser.add_argument('--end_date', required=True, help='End date for the prediction period (e.g., 2025-12-31).')
	parser.add_argument('--weekday_groups', required=True, help='Weekday groupings (e.g., {"Mon-Thu": [0,1,2,3], "Friday": [4]}).')
	parser.add_argument('--holiday_map', required=False, help='Optional holiday map (e.g., {"2024-12-25": 5}).', default=None)

	# Parse the arguments
	args = parser.parse_args()

	# Convert weekday_groups from string to dictionary
	weekday_groups = eval(args.weekday_groups)  # Using eval to convert the string to a dictionary. Use with caution.

	# Convert holiday_map from string to dictionary (if supplied)
	holiday_map = eval(args.holiday_map) if args.holiday_map else None

	# Define prediction period as tuple
	prediction_period = (args.start_date, args.end_date)



	repo_root = Path(__file__).parent.parent
	# Prefix the file paths with repo_root
	file_paths = [os.path.join(repo_root, file_path) for file_path in args.file_paths]

	# Call the prediction function with supplied arguments
	predicted_data = predict_next_year(
		file_paths=file_paths,
		prediction_period=prediction_period,
		weekday_groups=weekday_groups,
		holiday_map=holiday_map
	)

	# Display the first few predictions
	print(predicted_data.head())

if __name__ == "__main__":
	main()
