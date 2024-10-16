import pandas as pd
from pathlib import Path
import argparse
import os
from typing import List, Tuple, Dict, Optional

def load_historical_data(file_paths: List[str]) -> pd.DataFrame:
	"""
	Load historical time series data from multiple files.
	Each file should contain a time-indexed DataFrame.
	"""
	historical_data = pd.concat([pd.read_csv(file_path, parse_dates=[0], index_col=0) for file_path in file_paths])
	historical_data = historical_data.sort_index()
	return historical_data

def find_nearest_comparison_days(target_date: pd.Timestamp, historical_data: pd.DataFrame, group: str, weekday_groups: Dict[str, List[int]], num_days: int = 4) -> pd.Index:
	"""
	Find the nearest comparison days for the target date from the historical data.
	Ensure the comparison days belong to the same group (e.g., Mon-Thu).
	"""
	# Get the days that match the current group (e.g., Mon-Thu)
	comparison_days = historical_data[historical_data.index.weekday.isin(weekday_groups[group])]

	# Filter by calendar month and day for nearest match
	comparison_days = comparison_days.loc[(comparison_days.index.month == target_date.month) &
										  (comparison_days.index.day == target_date.day)]

	# Sort by the absolute difference in years and take the closest `num_days`
	nearest_days = comparison_days.index.to_series().apply(lambda x: abs((x - target_date).days)).nsmallest(num_days).index

	return nearest_days

def apply_holiday_map(date: pd.Timestamp, holiday_map: Optional[Dict[pd.Timestamp, int]]) -> int:
	"""
	Apply holiday map to treat special days as a different weekday.
	"""
	if date in holiday_map:
		return holiday_map[date]
	return date.weekday()

def get_weekday_group(weekday: int, weekday_groups: Dict[str, List[int]]) -> str:
	"""
	Return the group label based on the supplied weekday groupings.
	"""
	for group, days in weekday_groups.items():
		if weekday in days:
			return group
	raise ValueError(f'weekday {weekday} not in {weekday_groups}')

def predict_next_year(file_paths: List[str], prediction_period: Tuple[str, str], weekday_groups: Dict[str, List[int]], freq: str, holiday_map: Optional[Dict[pd.Timestamp, int]] = None) -> pd.DataFrame:
	"""
	Predict the next year's values based on the historical data and supplied parameters.

	Parameters:
	- file_paths: List of file paths containing historical time series data
	- prediction_period: Tuple of start and end date for the prediction (e.g., ('2025-01-01', '2025-12-31'))
	- weekday_groups: Dictionary specifying how weekdays are grouped
	- freq: Frequency for prediction intervals (e.g., '15min' for 15 minutes, '60min' for 1 hour)
	- holiday_map: Dictionary of specific dates to be treated as another weekday (optional)

	Returns:
	- A DataFrame with predicted values for the specified period
	"""
	# Load multiple historical data files
	historical_data = load_historical_data(file_paths)

	predictions = []

	# Convert prediction period strings to Timestamps
	start_date, end_date = pd.Timestamp(prediction_period[0]), pd.Timestamp(prediction_period[1])

	for target_date in pd.date_range(start_date, end_date, freq=freq):

		# Adjust for holiday mapping if provided
		weekday = apply_holiday_map(target_date, holiday_map) if holiday_map else target_date.weekday()

		# Determine the group for the target date's weekday
		group = get_weekday_group(weekday, weekday_groups)

		# Find nearest comparison days based on calendar day and group
		comparison_days = find_nearest_comparison_days(target_date, historical_data, group, weekday_groups)

		# Take the average of the corresponding interval values from the comparison days
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
	parser.add_argument('--freq', required=False, help='Frequency for prediction intervals (e.g., 15T for 15 minutes, 60T for 1 hour).', default='15T')

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
		freq=args.freq,
		holiday_map=holiday_map
	)

	# Display the first few predictions
	print(predicted_data.head())

if __name__ == "__main__":
	main()
