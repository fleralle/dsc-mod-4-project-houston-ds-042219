import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
import pandas as pd
import statsmodels.api as sm

def find_na_in_df(df):
	for column in df.columns:
		num_na_values = df[column].isna().sum()
		if num_na_values > 0:
				print(column, num_na_values)

def find_duplicates_based_on_each_column(df):
	for column in df.columns:
		print(column, len(df[df.duplicated(subset=column)]))

def convert_row_to_df(row, row_index):
	new_df = pd.DataFrame()
	new_df['prices'] = row[7:].dropna().astype(float)
	return new_df

def convert_rows_to_df_dict(df):
	df_dict = {}
	for row_index, row in df.iterrows():
		region_id = row['RegionID']
		df_dict[region_id] = convert_row_to_df(row, row_index)
	return df_dict

def check_for_stationarity(ts):
	from statsmodels.tsa.stattools import adfuller
	result = adfuller(ts)
	statistic, pvalue = result[0], result[1]

	return statistic, pvalue

def get_ols_result(ts):
	X = ts.index - ts.index[0]
	X = np.array([x.days for x in X]).reshape(-1,1)
	X_with_constant = sm.add_constant(X)

	y = ts.values.reshape(-1,1)

	ols = sm.OLS(y, X_with_constant)
	ols_result = ols.fit()

	return ols_result

def get_slope_and_pvalue(ts):
	ols_result = get_ols_result(ts)
	slope = ols_result.params[1]
	pvalue = ols_result.pvalues[1]
	return slope, pvalue

def load_and_prepare_zillow_data(csv_filename="zillow_data.csv"):
	df = pd.read_csv(csv_filename) # Load data
	df["Country"] = ["USA"]*len(df) # Add country column
	df["RegionName"] = df["RegionName"].astype(str) # Convert RegionName to string
	df.drop(['RegionID', 'SizeRank'], axis=1, inplace=True) # drop irrelevant columns

	groupby_df = df.groupby(['Country', 'State', 'Metro', 'CountyName', 'City', 'RegionName']) # groupby

	transformed_groupby_df = groupby_df.mean().transform(lambda x: np.log(x)).T # log transform and transpose

	transformed_groupby_df.set_index(pd.to_datetime(transformed_groupby_df.index), inplace=True) # set datetime index

	return transformed_groupby_df

def perform_train_test_split(df, train_start_year = '2012', train_end_year = '2015', test_start_year = '2016', test_end_year = '2018'):
	# perform train-test split
	train_df = df[train_start_year:train_end_year]
	test_df = df[test_start_year:test_end_year]

	return train_df, test_df

def get_level_dicts(train_df):
	country_dict = {}
	state_dict = {}
	metro_dict = {}
	county_dict = {}
	city_dict = {}
	zipcode_dict = {}

	for country in train_df.columns.get_level_values(level=0).unique():
		ts = train_df[country].dropna(how='all').mean(axis=1)
		if len(ts) == 0:
			continue
		country_dict[country] = get_slope_and_pvalue(ts)
		
		for state in train_df[country].columns.get_level_values(level=0).unique():
			ts = train_df[country][state].dropna(how='all').mean(axis=1)
			if len(ts) == 0:
				continue
			state_dict[(country, state)] = get_slope_and_pvalue(ts)
						
			for metro in train_df[country][state].columns.get_level_values(level=0).unique():
				ts = train_df[country][state][metro].dropna(how='all').mean(axis=1)
				if len(ts) == 0:
					continue
				metro_dict[(country, state, metro)] = get_slope_and_pvalue(ts)
	
				for county in train_df[country][state][metro].columns.get_level_values(level=0).unique():
					ts = train_df[country][state][metro][county].dropna(how='all').mean(axis=1)
					if len(ts) == 0:
						continue
					county_dict[(country, state, metro, county)] = get_slope_and_pvalue(ts)
			
					for city in train_df[country][state][metro][county].columns.get_level_values(level=0).unique():
						ts = train_df[country][state][metro][county][city].dropna(how='all').mean(axis=1)
						if len(ts) == 0:
							continue
						city_dict[(country, state, metro, county, city)] = get_slope_and_pvalue(ts)
			
						for zipcode in train_df[country][state][metro][county][city].columns.get_level_values(level=0).unique():
							ts = train_df[country][state][metro][county][city][zipcode].dropna()
							if len(ts) == 0:
								continue
							zipcode_dict[(country, state, metro, county, city, zipcode)] = get_slope_and_pvalue(ts)
	return 	country_dict, state_dict, metro_dict, county_dict, city_dict, zipcode_dict

def get_good_zipcodes(train_df, country_dict, state_dict, metro_dict, county_dict, city_dict, zipcode_dict, pvalue_threshold = 1e-15, check_levels=["State", "Metro", "County", "City", "Zipcode"]):
	good_zipcodes = {}
	for country in train_df.columns.get_level_values(level=0).unique():
		country_mvalue, country_pvalue = country_dict[country]
		
		for state in train_df[country].columns.get_level_values(level=0).unique():
			try:
				mvalue, pvalue = state_dict[(country, state)]
			except KeyError:
				continue
			if (mvalue < country_mvalue or pvalue > pvalue_threshold) and "State" in check_levels:
				continue

			for metro in train_df[country][state].columns.get_level_values(level=0).unique():
				try:
					mvalue, pvalue = metro_dict[(country, state, metro)]
				except KeyError:
					continue
				if (mvalue < country_mvalue or pvalue > pvalue_threshold) and "Metro" in check_levels:
					continue
						
				for county in train_df[country][state][metro].columns.get_level_values(level=0).unique():
					try:
						mvalue, pvalue = county_dict[(country, state, metro, county)]
					except KeyError:
						continue
					if (mvalue < country_mvalue or pvalue > pvalue_threshold) and "County" in check_levels:
						continue
					
					for city in train_df[country][state][metro][county].columns.get_level_values(level=0).unique():
						try:
							mvalue, pvalue = city_dict[(country, state, metro, county, city)]
						except KeyError:
							continue
						if (mvalue < country_mvalue or pvalue > pvalue_threshold) and "City" in check_levels:
							continue

						for zipcode in train_df[country][state][metro][county][city].columns.get_level_values(level=0).unique():
							try:
								mvalue, pvalue = zipcode_dict[(country, state, metro, county, city, zipcode)]
							except KeyError:
								continue
							if (mvalue < country_mvalue or pvalue > pvalue_threshold) and "Zipcode" in check_levels:
								continue
							good_zipcodes[(country, state, metro, county, city, zipcode)] = zipcode_dict[(country, state, metro, county, city, zipcode)]

	return good_zipcodes

def sort_good_zipcodes(good_zipcodes):
	x = good_zipcodes
	sorted_zipcodes = [zipcode for zipcode in sorted(x.items(), key=lambda kv: kv[1][0], reverse=True) if not np.isnan(zipcode[1][0]) and not np.isnan(zipcode[1][1])]

	return sorted_zipcodes

def plot_good_zipcodes(train_df, test_df, country_dict, sorted_zipcodes, num_top_zipcodes_to_plot=5):
	colors = ['b', 'g', 'r', 'c', 'm']*10

	train_df['USA'].mean(axis=1).plot(style="k", label="USA", figsize=(10,6))
	test_df['USA'].mean(axis=1).plot(style="k.", label="")
	print(country_dict['USA'])
	for i in range(num_top_zipcodes_to_plot):
		try:
			top_zipcode = sorted_zipcodes[i]
		except IndexError:
			continue
		print(top_zipcode)
		train_df[top_zipcode[0]].plot(style=f"{colors[i]}", label=top_zipcode[0][-1])
		test_df[top_zipcode[0]].plot(style=f"{colors[i]}.", label="")
	plt.legend()

def get_predicted_and_actual_returns(test_df, sorted_zipcodes, num_zipcodes_to_invest_in=5, investment_per_zipcode=1):
	num_days_to_invest = (test_df.index[-1] - test_df.index[0]).days

	total_predicted_returns = 0
	total_actual_returns = 0
	for i in range(num_zipcodes_to_invest_in):
		try:
			top_zipcode = sorted_zipcodes[i]
		except IndexError:
			continue
		mvalue = top_zipcode[1][0]
		predicted_return = investment_per_zipcode*np.exp(mvalue*num_days_to_invest)-investment_per_zipcode
		total_predicted_returns += predicted_return
		actual_effective_mvalue = (test_df[top_zipcode[0]][-1]-test_df[top_zipcode[0]][0])/num_days_to_invest
		actual_return = investment_per_zipcode*np.exp(actual_effective_mvalue*num_days_to_invest)-investment_per_zipcode
		total_actual_returns += actual_return
		# ts = test_df[top_zipcode[0]]
		# print(get_slope_and_pvalue(ts))

		#print(top_zipcode[0][-1], mvalue, actual_effective_mvalue)

	return total_predicted_returns, total_actual_returns
	
def test_train_test_years(df, check_levels=["State", "Metro", "County", "City", "Zipcode"], pvalue_threshold=1e-15, num_zipcodes_to_invest_in=5, investment_per_zipcode=1, train_start_year = '2012', train_end_year = '2015', test_start_year = '2016', test_end_year = '2018', num_shuffles_to_perform = 1000):
	train_df, test_df = perform_train_test_split(df, train_start_year=train_start_year, train_end_year=train_end_year, test_start_year=test_start_year, test_end_year=test_end_year)

	country_dict, state_dict, metro_dict, county_dict, city_dict, zipcode_dict = get_level_dicts(train_df)

	good_zipcodes = get_good_zipcodes(train_df, country_dict, state_dict, metro_dict, county_dict, city_dict, zipcode_dict, pvalue_threshold=pvalue_threshold, check_levels=check_levels)

	sorted_zipcodes = sort_good_zipcodes(good_zipcodes)

	plot_good_zipcodes(train_df, test_df, country_dict, sorted_zipcodes, num_top_zipcodes_to_plot=5)

	plt.figure()

	predicted_return, actual_return = get_predicted_and_actual_returns(test_df, sorted_zipcodes, num_zipcodes_to_invest_in=num_zipcodes_to_invest_in, investment_per_zipcode=investment_per_zipcode)
	print(predicted_return, actual_return)

	zipcodes = [x[0] for x in zipcode_dict]
	random_returns = []
	for i in range(num_shuffles_to_perform):
		random_zipcodes = list(zipcodes)
		np.random.shuffle(random_zipcodes)
		predicted_return, random_return = get_predicted_and_actual_returns(test_df, random_zipcodes, num_zipcodes_to_invest_in=num_zipcodes_to_invest_in, investment_per_zipcode=investment_per_zipcode)
		random_returns.append(random_return)
	plt.hist(random_returns)
	plt.vlines(x=actual_return, ymin=0, ymax=num_shuffles_to_perform/4, color='red')