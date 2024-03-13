from utils import *
import declare as dcl
import matplotlib
matplotlib.use('TkAgg')  # Or another backend as appropriate for your environment
import matplotlib.pyplot as plt

# first step:
#calculate the monthly vehicle volume of each service center
_,service_center_monthly_volumes_dict=calculate_service_centers_monthly_delivered_volume(dcl.service_center_delivered_df)


# second step:
#create service centers with determined number of technicians
service_centers=create_service_centers(number_of_service_centers=dcl.number_of_service_centers,max_working_hours_monthly=dcl.max_working_hours_service_center_in_month,pre_delivery_service_duration_mean=dcl.pre_delivery_service_duration_mean,pre_delivery_service_duration_variance=dcl.pre_delivery_service_duration_variance,post_delivery_arrival_mean=dcl.post_delivery_service_arrival_mean,post_delivery_arrival_variance=dcl.post_delivery_service_arrival_variance,post_delivery_service_durations_events_data=dcl.post_delivery_dervice_durations_data)
# log_objects_status('created service centers',service_centers)

#uncomment to print all attributes of the service centers
# for service_center in service_centers:
#     print(vars(service_center))
print(dcl.section)

#print upcoming vehicle volume of each service center
for service_center in service_centers:
    log_info(f"Monthly upcoming vehicle volume - Service Center {service_center.id}:{service_center_monthly_volumes_dict[f'Service Center {service_center.id}']}")
print(dcl.section)

# Initialize a list to store DataFrames from each iteration
all_performance_metrics = []


# Initialize a list to store DataFrames from each iteration
all_monthly_performance_metrics = []

for service_center in service_centers:
    for number_of_technicians in range(dcl.minimum_number_of_technicians, dcl.maximum_number_of_technicians+1):
        monthly_performance_report, _ = simulate_service_centers_performance(service_center=service_center, year=dcl.year, number_of_months=dcl.number_of_months, new_vehicles_monthly_volumes=service_center_monthly_volumes_dict[f'Service Center {service_center.id}'], number_of_technicians_to_hire=number_of_technicians, number_of_scenarios=dcl.number_of_simulation_scenarios)

        # Initialize a dictionary to store the average monthly performance metrics
        average_monthly_performance_metrics = {}

        for key in dcl.reportings_monthly_performance_metrics_keys:
            # Compute the mean across scenarios (axis=0) for each month
            average_monthly_performance_metrics[key] = np.mean(monthly_performance_report[key], axis=0)

        # Create a DataFrame for the current iteration
        df_current_monthly = pd.DataFrame(average_monthly_performance_metrics)
        df_current_monthly['Month'] = np.arange(1, dcl.number_of_months + 1)

        # Add identifiers for Service Center ID and Number of Technicians
        df_current_monthly['Service Center ID'] = service_center.id
        df_current_monthly['Number of Technicians'] = number_of_technicians

        # Append the current DataFrame to the list
        all_monthly_performance_metrics.append(df_current_monthly)
        # plot_summary_results(service_center=service_center,monthly_performance_report=monthly_performance_report,number_of_technicians=number_of_technicians)

# Concatenate all DataFrames in the list into a final DataFrame
df_final_monthly_performance_metrics = pd.concat(all_monthly_performance_metrics)

# Reset the index if you want the final DataFrame to have a continuous index
df_final_monthly_performance_metrics.reset_index(drop=True, inplace=True)

# Save the final DataFrame to a CSV file
df_final_monthly_performance_metrics.to_csv('all_service_centers_average_monthly_performance_metrics.csv')

