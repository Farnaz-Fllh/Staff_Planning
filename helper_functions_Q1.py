from scipy import stats
import pandas as pd
import numpy as np
import random
from scipy.stats import truncnorm
from thefuzz import process
np.random.seed(70)
import declare as dcl
section = '=' * 120
import matplotlib.pyplot as plt
from scipy.stats import norm


def set_pre_delivery_service_duration():
    """
    Generate a random service duration for pre-delivery service.

    Returns:
        int: The randomly generated service duration.
    """
    # ------------ generate service duration -------------
    std_dev = np.sqrt(dcl.pre_delivery_service_duration_variance)
    # Draw a random number from a normal distribution
    lower_bound = 0
    upper_bound = np.inf  # Using np.inf as there's no upper limit
    # Convert bounds to z-scores
    a = (lower_bound - dcl.pre_delivery_service_duration_mean) / std_dev
    b = (upper_bound - dcl.pre_delivery_service_duration_mean) / std_dev
    # Generate one random number
    service_duration = np.ceil(truncnorm.rvs(
        a, b, loc=dcl.pre_delivery_service_duration_mean, scale=std_dev))
    return service_duration
def set_post_delivery_service_duration():
    """
    Sets the duration of the post delivery service.

    Returns:
    - The randomly chosen duration from the post delivery service durations data.
    """
    return np.random.choice(dcl.post_delivery_dervice_durations_data)
def calculate_service_centers_monthly_delivered_volume(service_center_delivered_volume):
    """
    Calculate the monthly demand per service center from the provided data.
    If the demand does not exist for a month, it is considered to be zero.

    Parameters:
    - data: A dictionary containing 'Date', 'Distance to Service Center 1 (miles)',
      'Distance to Service Center 2 (miles)', 'Customer Location', and
      'Customer Delivered Volume'.

    Returns:
    - A pandas DataFrame containing the monthly demand for each service center.
    """
    # Convert the provided data into a pandas DataFrame
    df = pd.DataFrame(service_center_delivered_volume)

    # address the naming dicsrepencies of customer locations

    # Assuming these are the correct names based on the given dataset
    correct_locations = ['Los Angeles, CA', 'Irvince, CA', 'Long Beach, CA']

    # Applying fuzzy matching to correct the names
    df['Corrected Location'] = df['Customer Location'].apply(
        lambda x: process.extractOne(x, correct_locations)[0])

    # Drop the original 'Customer Location' column
    df.drop('Customer Location', axis=1, inplace=True)

    # Rename 'Corrected Location' to 'Customer Location'
    df.rename(
        columns={'Corrected Location': 'Customer Location'}, inplace=True)

    # Convert 'Date' to datetime and extract the month
    df['Date'] = pd.to_datetime(df['Date'])
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month

    # Determine the nearest service center
    df['Nearest Service Center'] = df.apply(lambda x: 'Service Center 1' if x['Distance to Service Center 1 (miles)']
                                            < x['Distance to Service Center 2 (miles)'] else 'Service Center 2', axis=1)

    # Group by Year, Month, and Nearest Service Center, then sum the delivered volume
    monthly_volume_per_service_center = df.groupby(['Year', 'Month', 'Nearest Service Center'])[
        'Customer Delivered Volume'].sum().reset_index()
    monthly_volume_per_service_center.columns = [
        'Year', 'Month', 'Service Center', 'Total Delivered Volume']
    # Creating a range of months to ensure every month is covered
    months = pd.date_range(
        start=f"{df['Year'].min()}-01-01", end=f"{df['Year'].max()}-12-31", freq='MS')

    # Creating a dataframe that has an entry for each service center for each month
    all_months_df = pd.DataFrame({'Date': pd.to_datetime(months)})
    all_months_df['key'] = 1
    service_centers = pd.DataFrame(
        {'Service Center': ['Service Center 1', 'Service Center 2'], 'key': 1})
    all_months_combination = pd.merge(
        all_months_df, service_centers, on='key').drop('key', axis=1)

    # Extracting year and month for merging
    all_months_combination['Year'] = all_months_combination['Date'].dt.year
    all_months_combination['Month'] = all_months_combination['Date'].dt.month

    # Merging with the monthly_volume_per_service_center to include zero demand where applicable
    monthly_demand_full = pd.merge(all_months_combination, monthly_volume_per_service_center, on=[
                                   'Year', 'Month', 'Service Center'], how='left').fillna(0)  # Fill missing demand with zeros

    # Dropping the Year column to match the requested final format
    service_centers_monthly_volumes_df = monthly_demand_full.drop(
        ['Year'], axis=1).sort_values(by=['Service Center', 'Date'])

    # Creating the dictionary
    service_center_monthly_volumes_dict = {}
    for center in service_centers_monthly_volumes_df['Service Center'].unique():
        volumes = service_centers_monthly_volumes_df.loc[
            service_centers_monthly_volumes_df['Service Center'] == center, 'Total Delivered Volume'].tolist()
        service_center_monthly_volumes_dict[center] = volumes

    service_center_monthly_volumes_dict = {key: [int(
        num) for num in value] for key, value in service_center_monthly_volumes_dict.items()}

    return service_centers_monthly_volumes_df, service_center_monthly_volumes_dict
def total_pre_deliverty_service_time(pre_delivery_vehicles):
    """
    Calculates the total service time for pre-delivery vehicles.

    Parameters:
    pre_delivery_vehicles (int): The number of pre-delivery vehicles.

    Returns:
    int: The total service time for all pre-delivery vehicles.
    """
    total_service_time = 0
    for _ in range(pre_delivery_vehicles):
        total_service_time += set_pre_delivery_service_duration()
    return total_service_time

def total_post_deliverty_service_time(post_delivery_vehicles):
    """
    Calculates the total service time for post delivery vehicles.

    Args:
        post_delivery_vehicles (int): The number of post delivery vehicles.

    Returns:
        float: The total service time for post delivery vehicles.
    """
    rands = [random.random() for i in range(post_delivery_vehicles)]
    total_vehicle_failed = sum(1 for num in rands if num < dcl.post_delivery_service_arrival_mean)
    total_service_time = 0
    for _ in range(total_vehicle_failed):
        total_service_time += set_post_delivery_service_duration()
    return total_service_time, total_vehicle_failed

def calculate_service_times(service_center_volume):
    """
    Simulates the service times for each month based on the demand data.

    Parameters:
    service_center_volume (list): A list of integers representing the volume of service center for each month.

    Returns:
    tuple: A tuple containing three lists - monthly_pre_delivery_time, monthly_post_delivery_time, and total_service_time.
         - monthly_pre_delivery_time: A list of floats representing the pre-delivery service time for each month.
         - monthly_post_delivery_time: A list of floats representing the post-delivery service time for each month.
         - total_service_time: A list of floats representing the total service time for each month.
    """
    total_vehicles_volume = 0
    pre_delivery_service_time = []
    post_delivery_service_time = []
    total_service_time = []
    total_failed_vehicles = []
    num_months = len(service_center_volume)
    for month in range(1, num_months+1):
        if month == 1:
            volume = service_center_volume[month-1]
            total_vehicles_volume += volume
            pre_delivery_service_time.append(
                total_pre_deliverty_service_time(volume))
            post_delivery_service_time.append(0)
            total_failed_vehicles.append(0)
        else:
            volume = service_center_volume[month-1]
            total_vehicles_volume += volume
            pre_delivery_service_time.append(total_pre_deliverty_service_time(volume))
            post_delivery_time_per_month, total_vehicle_failed = total_post_deliverty_service_time(total_vehicles_volume)
            post_delivery_service_time.append(post_delivery_time_per_month)
            total_failed_vehicles.append(total_vehicle_failed)
            

    total_service_time = [pre_delivery_service_time[i] +
                          post_delivery_service_time[i] for i in range(num_months)]

    return pre_delivery_service_time, post_delivery_service_time, total_service_time, total_failed_vehicles

def plot_multiple_lists(numbers1, numbers2, numbers3):
    """
    Plots multiple lists on a single figure with subplots.

    Parameters:
    - numbers1 (list): List of numbers for the first subplot.
    - numbers2 (list): List of numbers for the second subplot.
    - numbers3 (list): List of numbers for the third subplot.

    Returns:
    None
    """

    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

    plt.figure(figsize=(15, 6))

    plt.subplot(1, 3, 1)
    plt.plot(months, numbers1, marker='o', color='b', linestyle='-')
    plt.title('pre delivery')
    plt.xlabel('Months')
    plt.ylabel('service time')
    plt.grid(True)

    plt.subplot(1, 3, 2)
    plt.plot(months, numbers2, marker='o', color='r', linestyle='-')
    plt.title('post delivery')
    plt.xlabel('Months')
    plt.ylabel('service time')
    plt.grid(True)

    plt.subplot(1, 3, 3)
    plt.plot(months, numbers3, marker='o', color='g', linestyle='-')
    plt.title('Total')
    plt.xlabel('Months')
    plt.ylabel('service time')
    plt.grid(True)

    plt.tight_layout()
    
    
def boostraping(service_center_volume, num_sample=10):
    """
    Perform boostraping to calculate service times.

    Parameters:
    - service_center_volume (int): The volume of the service center.
    - num_sample (int): The number of samples to generate (default: 10).

    Returns:
    - sampling_total_service_time (numpy.ndarray): Array of total service times.
    - sampling_pre_delivery_time (numpy.ndarray): Array of pre-delivery times.
    - sampling_post_delivery_time (numpy.ndarray): Array of post-delivery times.
    """
    sampling_total_service_time = []
    sampling_pre_delivery_time = []
    sampling_post_delivery_time = []
    sampling_total_failed_vehicles = []

    for _ in range(num_sample):
        pre_delivery_time, post_delivery_time, total_service_times,total_failed_vehicles = calculate_service_times(
            service_center_volume)
        sampling_total_service_time.append(total_service_times)
        sampling_pre_delivery_time.append(pre_delivery_time)
        sampling_post_delivery_time.append(post_delivery_time)
        sampling_total_failed_vehicles.append(total_failed_vehicles)

    return np.array(sampling_total_service_time), np.array(sampling_pre_delivery_time), np.array(sampling_post_delivery_time), np.array(sampling_total_failed_vehicles)

    return np.array(sampling_total_service_time), np.array(sampling_pre_delivery_time), np.array(sampling_post_delivery_time)

def calculate_and_plot_average_failed_vehicles(data):
    data = np.array(data)
    # Calculate the average of each column
    column_averages = np.mean(data, axis=0)
    
    # Extract the average failed vehicles for all 12 months
    short_month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    month_averages = column_averages[:12]  # Assuming the data has 12 columns for each month
    
    # Plot the average failed vehicles for all 12 months
    plt.figure(figsize=(12, 6))
    plt.bar(short_month_names, month_averages, color='skyblue')
    plt.xlabel('Month')
    plt.ylabel('Average Failed Vehicles')
    plt.title('Average Failed Vehicles for Each Month')
    plt.show()

def calculate_confidence_interval(data_array, confidence_level=0.95):
    """
    Calculate the confidence intervals for each month in the data array.

    Parameters:
    - data_array (numpy.ndarray): The input data array with shape (num_samples, num_months).
    - confidence_level (float, optional): The desired confidence level (default: 0.95).

    Returns:
    - confidence_intervals (list): A list of tuples representing the confidence intervals for each month.
        Each tuple contains the lower and upper bounds of the confidence interval.

    Example:
    >>> data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    >>> calculate_confidence_interval(data)
    [(0.07, 6.93), (1.07, 7.93), (2.07, 8.93)]
    """
    num_samples, num_months = data_array.shape
    confidence_intervals = []

    z_score = norm.ppf(1 - (1 - confidence_level) / 2)

    for i in range(num_months):
        sample_mean = np.mean(data_array[:, i])
        # Use sample standard deviation
        sample_std = np.std(data_array[:, i], ddof=1)
        margin_of_error = z_score * (sample_std / np.sqrt(num_samples))

        lower_bound = sample_mean - margin_of_error
        upper_bound = sample_mean + margin_of_error

        confidence_intervals.append(
            (round(lower_bound, 2), round(upper_bound, 2)))

    return confidence_intervals

def calculate_confidence_interval_and_technicians(data_array, confidence_level=0.95, technician_capacity=168):
    """
    Calculate the confidence intervals for each month in the data array and determine the minimum
    number of technicians required based on both the lower and upper bounds of these intervals,
    returning the results in a DataFrame.

    Parameters:
    - data_array (numpy.ndarray): The input data array with shape (num_samples, num_months).
    - confidence_level (float, optional): The desired confidence level (default: 0.95).
    - technician_capacity (int, optional): The monthly capacity of each technician in hours (default: 168).

    Returns:
    - pd.DataFrame: DataFrame containing months, confidence intervals, minimum number of technicians required based on lower bounds,
                    and minimum number of technicians required based on upper bounds.
    """
    num_samples, num_months = data_array.shape
    months = [f'Month {i+1}' for i in range(num_months)]
    confidence_intervals = []
    technicians_needed_lower = []
    technicians_needed_upper = []

    z_score = norm.ppf(1 - (1 - confidence_level) / 2)

    for i in range(num_months):
        sample_mean = np.mean(data_array[:, i])
        sample_std = np.std(data_array[:, i], ddof=1)
        margin_of_error = z_score * (sample_std / np.sqrt(num_samples))

        lower_bound = sample_mean - margin_of_error
        upper_bound = sample_mean + margin_of_error

        # Calculate minimum technicians based on the lower bound
        technicians_lower = np.ceil(lower_bound / technician_capacity)
        technicians_needed_lower.append(int(technicians_lower))
        
        # Calculate minimum technicians based on the upper bound
        technicians_upper = np.ceil(upper_bound / technician_capacity)
        technicians_needed_upper.append(int(technicians_upper))

        confidence_intervals.append((round(lower_bound, 2), round(upper_bound, 2)))

    # Creating DataFrame
    results_df = pd.DataFrame({
        'Month': months,
        'Confidence Interval (CI)': confidence_intervals,
        'Min Technicians Needed (Lower Bound CI)': technicians_needed_lower,
        'Min Technicians Needed (Upper Bound CI)': technicians_needed_upper
    })

    return results_df

def visualize_demand_statistics(data):
    def demand_statistics(data):
        # Calculate mean, median, minimum, and maximum demand for each month
        mean_demand = np.mean(data, axis=0)
        median_demand = np.median(data, axis=0)
        min_demand = np.min(data, axis=0)
        max_demand = np.max(data, axis=0)

        # Create a dictionary to store the statistics for each month
        statistics = {
            'Month': np.arange(1, 13),
            'Mean Demand': mean_demand,
            'Median Demand': median_demand,
            'Min Demand': min_demand,
            'Max Demand': max_demand
        }

        return statistics

    result = demand_statistics(data)

    # Create subplots for each month's statistics using box plots
    fig, axs = plt.subplots(2, 6, figsize=(18, 6))

    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

    for i in range(12):
        row = i // 6
        col = i % 6

        axs[row, col].boxplot([data[:, i]])
        axs[row, col].set_title(month_names[i])
        axs[row, col].set_xticklabels(['service time'])

    plt.tight_layout()
    plt.show()

def plot_post_delivery_service_durations(post_service_duration_df):
    import seaborn as sns

    # Determine the full range of service durations (from min to max)
    min_duration = post_service_duration_df['Service Duration'].min()
    max_duration = post_service_duration_df['Service Duration'].max()
    full_range_durations = range(min_duration, max_duration + 1)

    # Creating a new DataFrame to ensure all durations are represented
    full_duration_df = pd.DataFrame(full_range_durations, columns=['Service Duration']).merge(
        post_service_duration_df['Service Duration'].value_counts().rename('Count'), left_on='Service Duration',
        right_index=True, how='left').fillna(0)

    # Plotting
    plt.figure(figsize=(12, 6))
    sns.barplot(x='Service Duration', y='Count', data=full_duration_df, palette='viridis')
    plt.xticks(range(len(full_range_durations)), full_range_durations)
    plt.title('Histogram of Service Durations')
    plt.xlabel('Serive Duration [hour]')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()

    
def plot_post_delivery_service_PMF_CDF(post_delivery_service_df):
    # Re-import necessary libraries and re-define the data since the execution state was reset
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    df = post_delivery_service_df

    # Calculate PMF (Probability Mass Function) for the discrete service durations
    pmf = df['Service Duration'].value_counts().sort_index() / len(df)
    pmf.index = pmf.index.astype(int)  # Ensure the index is integer for plotting

    # Calculate CDF (Cumulative Distribution Function) from the PMF
    cdf = pmf.cumsum()

    # Plotting PMF and CDF
    fig, ax1 = plt.subplots(figsize=(12, 6))

    color = 'tab:blue'
    ax1.set_xlabel('Service Duration (hours)')
    ax1.set_ylabel('PMF', color=color)
    ax1.bar(pmf.index, pmf, color=color, alpha=0.6, label='PMF')
    ax1.tick_params(axis='y', labelcolor=color)

    # Instantiate a second axes that shares the same x-axis
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('CDF', color=color)
    ax2.plot(cdf.index, cdf, color=color, marker='o', linestyle='-', linewidth=2, markersize=5, label='CDF')
    ax2.tick_params(axis='y', labelcolor=color)

    # Title and show plot
    fig.tight_layout()
    plt.title('PMF and CDF of Service Durations')
    plt.grid(True)
    plt.show()
    return pmf,cdf

def calculate_summary_statistics_post_delivery_service_durations(pmf,cdf):
      # Creating a DataFrame with Service Duration, PMF, and CDF
    distribution_df = pd.DataFrame({
        'Service Duration': pmf.index,
        'PMF': pmf.values,
        'CDF': cdf.values
    })
   
        
        # Calculate the expected service duration as the sum of product of service durations and their probabilities (PMF)
    expected_service_duration = (distribution_df['Service Duration'] * distribution_df['PMF']).sum()

    # Variance
    variance = ((distribution_df['Service Duration'] - expected_service_duration) ** 2 * distribution_df['PMF']).sum()

    # Standard Deviation
    standard_deviation = variance ** 0.5

    # Skewness
    skewness = ((distribution_df['Service Duration'] - expected_service_duration) ** 3 * distribution_df['PMF']).sum() / (standard_deviation ** 3)

    # Kurtosis
    kurtosis = ((distribution_df['Service Duration'] - expected_service_duration) ** 4 * distribution_df['PMF']).sum() / (standard_deviation ** 4) - 3  # Excess kurtosis

    # Calculating quartiles
    Q1 = distribution_df['Service Duration'].quantile(0.25)
    Q2 = distribution_df['Service Duration'].quantile(0.5)  # This is also the median
    Q3 = distribution_df['Service Duration'].quantile(0.75)

    # Calculating Interquartile Range (IQR)
    IQR = Q3 - Q1

    # Consolidating all metrics into a single dictionary for a condensed summary
    condensed_summary = {
        "Mean (Expected Service Duration)": expected_service_duration,
        "Variance": variance,
        "Standard Deviation": standard_deviation,
        "Skewness": skewness,
        "Kurtosis": kurtosis,
        "25th Percentile (Q1)": Q1,
        "Median (Q2)": Q2,
        "75th Percentile (Q3)": Q3,
        "Interquartile Range (IQR)": IQR
    }
    print(condensed_summary)
    
def gamma_weibull_goodness_of_fit_test(df):
    
    from scipy import stats
    import numpy as np

    # Extracting the service duration data
    service_durations = df['Service Duration'].values

    # Fit the Gamma distribution to the data
    gamma_params = stats.gamma.fit(service_durations)
    gamma_a, gamma_loc, gamma_scale = gamma_params
    gamma_rv = stats.gamma(gamma_a, loc=gamma_loc, scale=gamma_scale)

    # Fit the Weibull distribution to the data
    weibull_params = stats.weibull_min.fit(service_durations)
    weibull_c, weibull_loc, weibull_scale = weibull_params
    weibull_rv = stats.weibull_min(weibull_c, loc=weibull_loc, scale=weibull_scale)

    # Evaluate the goodness-of-fit using Kolmogorov-Smirnov test
    ks_stat_gamma, ks_pval_gamma = stats.kstest(service_durations, gamma_rv.cdf)
    ks_stat_weibull, ks_pval_weibull = stats.kstest(service_durations, weibull_rv.cdf)

    fit_results = {
        'Gamma': {'KS Statistic': ks_stat_gamma, 'p-value': ks_pval_gamma},
        'Weibull': {'KS Statistic': ks_stat_weibull, 'p-value': ks_pval_weibull}
    }

    print(fit_results)
    


    
