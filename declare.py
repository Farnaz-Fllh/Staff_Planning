import numpy as np
import pandas as pd
import time

#Read the data from the CSV file as a dataframe
service_center_delivered_df = pd.read_excel('DS Sample Data.xlsx', sheet_name='Service_Center_Delivered')
post_delivery_service_duration_events_df = pd.read_excel('DS Sample Data.xlsx', sheet_name='Service_Center_Times')
post_delivery_dervice_durations_data = post_delivery_service_duration_events_df['Service Duration']
#-------------------------------------#
list_of_service_centers=['Service Center 1','Service Center 2']
pre_delivery_service_type_str='pre-delivery'
post_delivery_service_type_str='post-delivery'
list_of_service_types=['pre-delivery','post-delivery']
#--------------------parameters----------------------------#
number_of_service_centers=2
minimum_number_of_technicians=1
maximum_number_of_technicians=5
pre_delivery_service_duration_mean=2
pre_delivery_service_duration_variance=1
post_delivery_service_arrival_mean=0.25
post_delivery_service_arrival_variance=0.2
#---------------------------------------------
year=2023
number_of_months=12
max_working_hours_service_center_in_month=168
max_working_hours_of_technicians_in_month=168
number_of_simulation_scenarios=50
#--------------------logging related parameters----------------------
start_time=time.time()
section='='*120
disable_info_log=True
disable_objects_status_info_log=True
disable_print_summary_metrics=True
#--------------------For reporting ------------------------------------
number_of_services_str='number of services'
number_of_pre_delivery_services_str='number of pre-delivery services'
number_of_post_delivery_services_str='number of post-delivery services'

# Tracking fulfilled services
number_of_fulfilled_services_str='number of fulfilled services'
number_of_fulfilled_pre_delivery_services_str='number of fulfilled pre-delivery services'
number_of_fulfilled_post_delivery_services_str='number of fulfilled post-delivery services'

# Tracking fulfilled services
duration_of_fulfilled_services_str='duration of fulfilled services'
duration_of_fulfilled_pre_delivery_services_str='duration of fulfilled pre-delivery services'
duration_of_fulfilled_post_delivery_services_str='duration of fulfilled post-delivery services'

#elapsed time from recieving the service request until the service gets started
queue_waiting_time_str='average waiting time in queue' #sum(start_service_time)
pre_delivery_service_demands_duration_str='total pre-delivery service duration'
post_delivery_service_demands_duration_str='total post_delivery service duration'
total_service_demands_duration_str='total service demands duration'
backlog_service_demand_durations_str='service demand duration backlog'
technicians_worked_hours_str='number of worked hours in a month'  #total worked hours of technicians/168*number of technicians

#------------------------------------------------------------------------------------------------------------
reportings_monthly_performance_metrics_keys=[number_of_services_str,number_of_pre_delivery_services_str,number_of_post_delivery_services_str,number_of_fulfilled_services_str,number_of_fulfilled_pre_delivery_services_str,
                number_of_fulfilled_post_delivery_services_str,queue_waiting_time_str,technicians_worked_hours_str,pre_delivery_service_demands_duration_str,post_delivery_service_demands_duration_str,total_service_demands_duration_str,duration_of_fulfilled_services_str,duration_of_fulfilled_pre_delivery_services_str,duration_of_fulfilled_post_delivery_services_str,backlog_service_demand_durations_str]

reportings_comulative_performance_metrics_keys=[number_of_services_str,number_of_pre_delivery_services_str,number_of_post_delivery_services_str,number_of_fulfilled_services_str,number_of_fulfilled_pre_delivery_services_str,
                number_of_fulfilled_post_delivery_services_str,technicians_worked_hours_str]


plot_monthly_performance_metrics_keys=[pre_delivery_service_demands_duration_str,post_delivery_service_demands_duration_str,total_service_demands_duration_str,
                                       backlog_service_demand_durations_str,duration_of_fulfilled_services_str,duration_of_fulfilled_pre_delivery_services_str,duration_of_fulfilled_post_delivery_services_str]

#-----------------------------------------------------------------------
reporting_comulative_performance_metrics={key:np.zeros((number_of_simulation_scenarios,number_of_months)) for key in reportings_comulative_performance_metrics_keys}
reporting_monthly_performance_metrics={key:np.zeros((number_of_simulation_scenarios,number_of_months)) for key in reportings_monthly_performance_metrics_keys}
