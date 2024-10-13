import pandas as pd
import numpy as np
from datetime import timedelta

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

def predict_next_year(historical_data, prediction_period, holiday_map=None):
	"""
	Predict the next year's values based on the historical data.
	"""
	predictions = []

	for target_date in pd.date_range(prediction_period[0], prediction_period[1], freq='15T'):

		# Adjust for holiday mapping if provided
		weekday = apply_holiday_map(target_date, holiday_map) if holiday_map else target_date.weekday()

		# Determine the nearest comparison days based on weekday groups
		if weekday in range(0, 4):  # Monday to Thursday
			group = 'Mon-Thu'
		elif weekday == 4:  # Friday
			group = 'Friday'
		elif weekday == 5:  # Saturday
			group = 'Saturday'
		else:  # Sunday
			group = 'Sunday'

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

# Historical data setup (time-indexed DataFrame with 15-minute intervals)
historical_data = pd.DataFrame(
	{'value': np.random.randn(35040)},  # Assume we have data for 1 year (365 days * 24 * 4 = 35040 intervals)
	index=pd.date_range(start='2022-01-01', periods=35040, freq='15T')
)

# Holiday map setup (date -> weekday, e.g., 5 means treating as Saturday)
holiday_map = {pd.Timestamp('2024-12-25'): 5}  # Treat Christmas as Saturday

# Predict for next year (2025)
prediction_period = (pd.Timestamp('2025-01-01'), pd.Timestamp('2025-12-31'))

predicted_data = predict_next_year(historical_data, prediction_period, holiday_map)

print(predicted_data.head())
