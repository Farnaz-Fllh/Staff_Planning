from scipy import stats
import pandas as pd
import numpy as np
import random
from scipy.stats import truncnorm
from thefuzz import process
import declare as dcl
import matplotlib.pyplot as plt
import seaborn as sns


# np.random.seed(70)
section = '=' * 120
#-----------------------------------------------------------------------------------------
def log_info(info_str):
    dot = '.' * 5
    if not dcl.disable_info_log:
        # print(line)
        print(f"{dot} elapsed-time: {round(dcl.time.time()-dcl.start_time,2)}| Info log: "+info_str +f" {'':<2}")
        # print(line)
def log_objects_status(info_str,val):
    line = '-' * 120
    if not dcl.disable_objects_status_info_log:
        print(line)
        print(f"{'':<50}status info: " + info_str + f" {'':<50}")
        if type(val)==list:
            for obj in val:
                print(obj.all_attributes())
        print(line)

def log_summary_metrics(month,monthly_summary_metrics,comulative_summary_metrics):
    line='^'*120
    print(line)
    print(f"{'':<50} (Date: {month}-{dcl.year}) service center monthly performance metrics  {'':<50}")
    for key, value in monthly_summary_metrics.items():
        print(f"{key}:")
        print(np.array2string(value, formatter={'float_kind': lambda x: "%.2f" % x}))
        print()  # Print a newline for spacing between entries
    print('.'*120)
    print(f"{'':<50} service center comulative summary metrics - from 1-{dcl.year} to {month}-{dcl.year} {'':<50}")
    for key, value in comulative_summary_metrics.items():
        print(f"{key}:")
        print(np.array2string(value, formatter={'float_kind': lambda x: "%.2f" % x}))
        print()  # Print a newline for spacing between entries
    print(line)
def plot_summary_results(service_center,monthly_performance_report,number_of_technicians):

    for key in monthly_performance_report.keys():
                if key in dcl.plot_monthly_performance_metrics_keys:
                    for month in range(dcl.number_of_months):
                        plt.figure(figsize=(10, 6))
                        sns.histplot(monthly_performance_report[key][:, month], kde=True)
                        plt.title(f'Distribution of {key} for Month {month + 1}')
                        plt.xlabel('service durations')
                        plt.ylabel('Frequency')
                        plt.grid(True)

                        # Save the figure
                        plt.savefig(f'plots\\service_center{service_center.id}\\service_center{service_center.id}-number of technicians{number_of_technicians}-{key}_Month_{month + 1}_monthly_report.png', bbox_inches='tight')
                        plt.close()  # Close the figure to free up memory
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
    df.rename(columns={'Corrected Location': 'Customer Location'}, inplace=True)

    # Convert 'Date' to datetime and extract the month
    df['Date'] = pd.to_datetime(df['Date'])
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month

    # Determine the nearest service center
    df['Nearest Service Center'] = df.apply(lambda x: 'Service Center 1' if x['Distance to Service Center 1 (miles)'] < x['Distance to Service Center 2 (miles)'] else 'Service Center 2', axis=1)

    # Group by Year, Month, and Nearest Service Center, then sum the delivered volume
    monthly_volume_per_service_center = df.groupby(['Year', 'Month', 'Nearest Service Center'])['Customer Delivered Volume'].sum().reset_index()
    monthly_volume_per_service_center.columns = ['Year', 'Month', 'Service Center', 'Total Delivered Volume']

    # Creating a range of months to ensure every month is covered
    months = pd.date_range(start=f"{df['Year'].min()}-01-01", end=f"{df['Year'].max()}-12-31", freq='MS')

    # Creating a dataframe that has an entry for each service center for each month
    all_months_df = pd.DataFrame({'Date': pd.to_datetime(months)})
    all_months_df['key'] = 1
    service_centers = pd.DataFrame({'Service Center': ['Service Center 1', 'Service Center 2'], 'key': 1})
    all_months_combination = pd.merge(all_months_df, service_centers, on='key').drop('key', axis=1)

    # Extracting year and month for merging
    all_months_combination['Year'] = all_months_combination['Date'].dt.year
    all_months_combination['Month'] = all_months_combination['Date'].dt.month

    # Merging with the monthly_volume_per_service_center to include zero demand where applicable
    monthly_demand_full = pd.merge(all_months_combination, monthly_volume_per_service_center, on=['Year', 'Month', 'Service Center'], how='left').fillna(0)  # Fill missing demand with zeros

    # Dropping the Year column to match the requested final format
    service_centers_monthly_volumes_df = monthly_demand_full.drop(['Year'], axis=1).sort_values(by=['Service Center', 'Date'])

    # Creating the dictionary
    service_center_monthly_volumes_dict = {}
    for center in service_centers_monthly_volumes_df['Service Center'].unique():
        volumes = service_centers_monthly_volumes_df.loc[
            service_centers_monthly_volumes_df['Service Center'] == center, 'Total Delivered Volume'].tolist()
        service_center_monthly_volumes_dict[center] = volumes

    return service_centers_monthly_volumes_df,service_center_monthly_volumes_dict
class Vehicle:
    def __init__(self, arrival_month,service_type,service_center_id):
        self.arrival_month=arrival_month #Arrival month
        self.service_type = service_type  # "pre-delivery" ,  "post-delivery" , "None"
        self.under_service=False
        self.start_service_time=None
        self.remaining_service_time=None
        self.service_duration = None
        self.assigned_technician=None
        self.service_center_id=service_center_id
        self.waiting_time=0


    #reseting the information of vehicle after being serviced
    def reset_after_serviced(self):
        self.arrival_month = None
        self.service_type = None  # "pre-delivery" ,  "post-delivery" , "None"
        self.under_service = False
        self.start_service_time = None
        self.remaining_service_time = None
        self.service_duration = None
        self.assigned_technician = None

    def start_getting_service(self,time,technician):
        self.assigned_technician=technician
        self.start_service_time=time
        self.under_service=True
        self.waiting_time+=time
        self.remaining_service_time=self.service_duration


    def request_post_delivery_service(self,request_month,service_duration):
        self.arrival_month=request_month
        self.service_type=dcl.post_delivery_service_type_str
        self.set_service_duration(service_duration)

    def set_service_duration(self, duration):
        self.service_duration = duration

    def get_assigned_technician_info(self):
        if self.assigned_technician!=None:
            return self.assigned_technician.id
        else: 'NA'

    def all_attributes(self):
        attributes = {"vehicle info: "
        "arrival month": self.arrival_month,  # Arrival month
        "service_type":self.service_type , # "pre-delivery" or "post-delivery"
        "start_service_time":self.start_service_time ,
        "remaining_service_time":self.remaining_service_time ,
        "service_duration":self.service_duration ,
        "assigned technician id":self.get_assigned_technician_info(),
        " assigned service center id":self.service_center_id
        }
        return attributes
class Technician:
    def __init__(self, id):
        self.id = id
        self.type='Full time'
        self.is_available=True
        self.worked_hours=0
        self.idle_time=np.ones(168)
        self.available_hours = 168  # Monthly available hours
        self.number_of_services_finished=0
        self.time_to_finish_service=None

    def reset_new_month(self):
        if self.is_available or self.time_to_finish_service==0:
            self.is_available = True
            self.worked_hours = 0
            self.available_hours = dcl.max_working_hours_of_technicians_in_month  # Monthly available hours
            self.number_of_services_finished = 0
            self.time_to_finish_service = None
        else:
            self.is_available = False
            self.worked_hours = 0
            self.available_hours = dcl.max_working_hours_of_technicians_in_month-self.time_to_finish_service  # Monthly available hours
            self.number_of_services_finished = 0

    def start_service_task(self,task_duration):
        self.is_available=False
        self.time_to_finish_service=task_duration
        self.available_hours-=task_duration

    def finished_service_task(self,task_duration):
        self.is_available=True
        self.number_of_services_finished+=1
        self.time_to_finish_service=0

    def still_working_on_vehicle(self):
        self.time_to_finish_service-=1
        self.worked_hours+=1

    def all_attributes(self):
        attributes = {
        "technician id":self.id,
        "is_available":self.is_available,
        "time to finish service":self.time_to_finish_service,
        "worked_hours": self.worked_hours,
        "available_hours":self.available_hours,  # Monthly available hours
        "number_of_services_finished": self.number_of_services_finished
        }
        return attributes
class ServiceCenter:
    _Time_tracker=0
    def __init__(self,id:int,max_working_hours:int,pre_delivery_service_duration_mean,pre_delivery_service_duration_variance,post_delivery_arrival_mean,post_delivery_arrival_variance,post_delivery_service_durations_events_data):
        self.id=id
        self.pre_delivery_service_duration_mean=pre_delivery_service_duration_mean
        self.pre_delivery_service_duration_variance=pre_delivery_service_duration_variance
        self.post_delivery_service_arrival_mean=post_delivery_arrival_mean
        self.post_delivery_service_arrival_variance=post_delivery_arrival_variance
        self.post_delivery_service_durations_data=post_delivery_service_durations_events_data

        self.technicians = []
        self.vehicles_under_service=[]
        self.queue = []
        self.max_working_hours=max_working_hours

        self.vehicles_serviced={(month+1,service_type):[]for month in range(dcl.number_of_months) for service_type in dcl.list_of_service_types}
        self.queue_waiting_time={month+1:[]for month in range(dcl.number_of_months)}
        self.service_demands_durations_this_month={dcl.pre_delivery_service_type_str:[],dcl.post_delivery_service_type_str:[]} #dict to save new services durations of the month
        self.service_demands_durations_fulfilled_this_month={(month+1,service_type):[]for month in range(dcl.number_of_months) for service_type in dcl.list_of_service_types} #to save the duration of services that are not fulfilled this month
        self.service_demands_durations_backlog={(month+1,service_type):[]for month in range(dcl.number_of_months) for service_type in dcl.list_of_service_types} #to save the duration of services that are not fulfilled this month
        # and are shifted to the next month

    def reset_monthly_service_demand_durations_list(self):
        self.service_demands_durations_this_month={dcl.pre_delivery_service_type_str:[],dcl.post_delivery_service_type_str:[]}
        self.service_demands_durations_fulfilled_this_month={(month+1,service_type):[]for month in range(dcl.number_of_months) for service_type in dcl.list_of_service_types} #to save the duration of services that are not fulfilled this month

    def reset_for_new_scenarios(self):
        self.queue=[]
        self.vehicles_under_service=[]
        self.vehicles_serviced={(month+1,service_type):[]for month in range(dcl.number_of_months) for service_type in dcl.list_of_service_types}
        self.queue_waiting_time={month+1:[]for month in range(dcl.number_of_months)}
        self.service_demands_durations_this_month={dcl.pre_delivery_service_type_str:[],dcl.post_delivery_service_type_str:[]} #dict to save new services durations of the month
        self.service_demands_durations_fulfilled_this_month={(month+1,service_type):[]for month in range(dcl.number_of_months) for service_type in dcl.list_of_service_types} #to save the duration of services that are not fulfilled this month
        self.service_demands_durations_backlog={(month+1,service_type):[]for month in range(dcl.number_of_months) for service_type in dcl.list_of_service_types} #to save the duration of services that are not fulfilled this month


    def add_technician(self, technician):
        self.technicians.append(technician)

    def add_vehicle_to_queue(self, vehicle):
        self.queue.append(vehicle)
        self.service_demands_durations_this_month[vehicle.service_type].append(vehicle.service_duration)


    def set_pre_delivery_service_duration(self):
        # ------------ generate service duration -------------
        std_dev = np.sqrt(self.pre_delivery_service_duration_variance)
        # Draw a random number from a normal distribution
        lower_bound = 0
        upper_bound = np.inf  # Using np.inf as there's no upper limit
        # Convert bounds to z-scores
        a = (lower_bound - self.pre_delivery_service_duration_mean) / std_dev
        b = (upper_bound - self.pre_delivery_service_duration_mean) / std_dev
        # Generate one random number
        service_duration = np.ceil(truncnorm.rvs(a, b, loc=self.pre_delivery_service_duration_mean, scale=std_dev))
        return service_duration

    def set_post_delivery_service_duration(self):
        return np.random.choice(self.post_delivery_service_durations_data)

    # New vehicles require pre_delivery services
    def generate_new_vehicles(self,month, pre_delivery_volume):
        vehicles_requested_for_pre_delivery_service=[]

        for _ in range(int(pre_delivery_volume)): # create vehicle
            vehicle = Vehicle(arrival_month=month,service_type=dcl.pre_delivery_service_type_str,service_center_id=self.id)

            #generate service duration
            service_duration=self.set_pre_delivery_service_duration()
            #-----------------------------------------------------
            vehicle.set_service_duration(service_duration)
            self.add_vehicle_to_queue(vehicle)
            vehicles_requested_for_pre_delivery_service.append(vehicle)
        log_info(f'< {len(vehicles_requested_for_pre_delivery_service)} > new vehicles with pre-delivery service requests joined the queue')
        return vehicles_requested_for_pre_delivery_service

    def generate_post_delivery_requests(self,month,vehicles_on_the_road):
        vehicles_requested_for_post_delivery_service=[]
        for vehicle in vehicles_on_the_road:
            if random.random()< self.post_delivery_service_arrival_mean:
                service_duration=self.set_post_delivery_service_duration()
                vehicle.request_post_delivery_service(request_month=month,service_duration=service_duration)
                vehicles_requested_for_post_delivery_service.append(vehicle)
                self.add_vehicle_to_queue(vehicle)
        log_info(f'(0-{month}-2023)|{len(vehicles_requested_for_post_delivery_service)} post delivery requests created')

        return vehicles_requested_for_post_delivery_service

    def simulate_month(self, month):
        number_of_serviced_vehicles_this_month = 0
        # beginning of the month
        time = 0
        while time < self.max_working_hours:

            temp_vehicles_serviced = []
            # first check if there is any vehicles under maintneance from previous time periods
            for vehicle in self.vehicles_under_service:
                # if a vehicle has been under service from previous months
                if vehicle.remaining_service_time>0:
                    vehicle.assigned_technician.still_working_on_vehicle() #time_to_finish_service-=1 & worked_hours+=1
                    vehicle.remaining_service_time -= 1

                if vehicle.remaining_service_time == 0:
                    log_info(f'technician {vehicle.assigned_technician.id} finished its {vehicle.service_type} service at t={time}')
                    number_of_serviced_vehicles_this_month += 1
                    self.vehicles_serviced[(month,vehicle.service_type)].append(vehicle)
                    self.service_demands_durations_fulfilled_this_month[(month,vehicle.service_type)].append(vehicle.service_duration)
                    vehicle.assigned_technician.finished_service_task(task_duration=vehicle.service_duration)
                    log_objects_status('updated technician info', vehicle.assigned_technician)
                    print(vehicle.assigned_technician.all_attributes())
                    vehicle.reset_after_serviced()
                    temp_vehicles_serviced.append(vehicle)

            # updating the under service and serviced vehicles lists
            for vehicle in temp_vehicles_serviced:
                self.vehicles_under_service.remove(vehicle)

            temp_vehicles_under_service = []
            if any(technician.is_available for technician in self.technicians):
                for vehicle in self.queue:
                    for technician in self.technicians:
                        if technician.available_hours >= vehicle.service_duration and technician.is_available and not vehicle.under_service:
                            log_info(f'technician {technician.id} starts a {vehicle.service_type} service at hour={time} for duration of {vehicle.service_duration} for the following vehicle:')
                            vehicle.start_getting_service(time=time, technician=technician)
                            self.queue_waiting_time[month].append(vehicle.waiting_time)
                            technician.start_service_task(task_duration=vehicle.service_duration)
                            temp_vehicles_under_service.append(vehicle)
                            log_info(f'vehicle service_type {vehicle.service_type} | remaining_service_time {vehicle.remaining_service_time} | waiting_time={vehicle.waiting_time} | service duration {vehicle.service_duration} | technician {vehicle.assigned_technician.id}')
                            break  # move to the next vehicle
            else:
                log_info(f"All technicians are busy @ hour = {time} and can't service the queue")

            for vehicle in temp_vehicles_under_service:
                self.queue.remove(vehicle)
                self.vehicles_under_service.append(vehicle)
            log_info(f'hour {time}-({month}-{dcl.year}) | queue size :{len(self.queue)}')

            # check to see if the queue is empty and all services are finished by technicians
            if len(self.queue) == 0 and len(self.vehicles_under_service) == 0:
                log_info(f'The queue is empty & All service requests are fullfilled @ hour = {time}')

                # set the time to the end of month
                time = dcl.max_working_hours_service_center_in_month - 1
            time += 1
            if time!=0:
                print('-' * 120)
                print(f"{'':<50}status info: service center {'':<50}")
                print(self.all_attributes(month))
                log_objects_status('technicians status', self.technicians)
            log_info('rolling forward to simulate the next hour')
            print('-'*120)
            log_info(f' Date:{month}-{dcl.year}| Hour {time}')
            print('-' * 120)

        #============================================================================
        for vehicle in self.vehicles_under_service:
            fulfilled_service=vehicle.service_duration-vehicle.remaining_service_time
            self.service_demands_durations_fulfilled_this_month[(month, vehicle.service_type)].append(fulfilled_service)
        #=======================================================================================
        self.current_time = time
        log_info(info_str=f'Date: {month}-{dcl.year} | Hour {time} | end of simulation of this month')
        #========================================================================================
    def monthly_summary_metrics(self,month,scenario,new_generated_vehicles_this_month,vehicles_post_delivery_service_this_month):
        #------------------------------------New Incoming service demand info of the month--------------------------------------------------
        #Number of service requests made this month
        dcl.reporting_monthly_performance_metrics[dcl.number_of_services_str][scenario-1][month-1] = len(new_generated_vehicles_this_month) + len(vehicles_post_delivery_service_this_month)
        dcl.reporting_monthly_performance_metrics[dcl.number_of_pre_delivery_services_str][scenario-1][month-1] = len(new_generated_vehicles_this_month)
        dcl.reporting_monthly_performance_metrics[dcl.number_of_post_delivery_services_str][scenario-1][month-1] = len(vehicles_post_delivery_service_this_month)

        # requested service demand durations of in the month
        dcl.reporting_monthly_performance_metrics[dcl.pre_delivery_service_demands_duration_str][scenario-1][month-1] = np.sum(self.service_demands_durations_this_month[dcl.pre_delivery_service_type_str])
        dcl.reporting_monthly_performance_metrics[dcl.post_delivery_service_demands_duration_str][scenario-1][month-1]= np.sum(self.service_demands_durations_this_month[dcl.post_delivery_service_type_str])
        dcl.reporting_monthly_performance_metrics[dcl.total_service_demands_duration_str][scenario-1][month-1] = np.sum(self.service_demands_durations_this_month[dcl.pre_delivery_service_type_str]) + np.sum(self.service_demands_durations_this_month[dcl.post_delivery_service_type_str])

        #---------------------------------------------------------------------------------------------------------------------------------------------
        #number of vehicles start getting service this month , including under service vehicles
        dcl.reporting_monthly_performance_metrics[dcl.number_of_fulfilled_services_str][scenario-1][month-1] = len(self.vehicles_serviced[(month,dcl.pre_delivery_service_type_str)]) + len(self.vehicles_serviced[(month,dcl.post_delivery_service_type_str)]) + len([1 for _ in self.vehicles_under_service])
        dcl.reporting_monthly_performance_metrics[dcl.number_of_fulfilled_pre_delivery_services_str][scenario-1][month-1]= len(self.vehicles_serviced[(month,dcl.pre_delivery_service_type_str)]) + len([1 for vehicle in self.vehicles_under_service if vehicle.service_type==dcl.pre_delivery_service_type_str])
        dcl.reporting_monthly_performance_metrics[dcl.number_of_fulfilled_post_delivery_services_str][scenario-1][month-1] =  len(self.vehicles_serviced[(month,dcl.post_delivery_service_type_str)]) + len([1 for vehicle in self.vehicles_under_service if vehicle.service_type==dcl.post_delivery_service_type_str])

        #service durations fulfilled , including under service vehicles
        dcl.reporting_monthly_performance_metrics[dcl.duration_of_fulfilled_services_str][scenario-1][month-1]= np.sum(self.service_demands_durations_fulfilled_this_month[(month,dcl.pre_delivery_service_type_str)]) + np.sum(self.service_demands_durations_fulfilled_this_month[(month,dcl.post_delivery_service_type_str)])
        dcl.reporting_monthly_performance_metrics[dcl.duration_of_fulfilled_pre_delivery_services_str][scenario-1][month-1] = np.sum(self.service_demands_durations_fulfilled_this_month[(month,dcl.pre_delivery_service_type_str)])
        dcl.reporting_monthly_performance_metrics[dcl.duration_of_fulfilled_post_delivery_services_str][scenario-1][month-1]=  np.sum(self.service_demands_durations_fulfilled_this_month[(month,dcl.post_delivery_service_type_str)])


        dcl.reporting_monthly_performance_metrics[dcl.technicians_worked_hours_str][scenario-1][month-1]=np.sum([technician.worked_hours for technician in self.technicians])
        #waiting time of the vehicles serviced
        dcl.reporting_monthly_performance_metrics[dcl.queue_waiting_time_str][scenario-1][month-1]=np.mean(self.queue_waiting_time[month])


        #requested service demand durations of in the month
        dcl.reporting_monthly_performance_metrics[dcl.pre_delivery_service_demands_duration_str][scenario-1][month-1]=np.sum(self.service_demands_durations_this_month[dcl.pre_delivery_service_type_str])
        dcl.reporting_monthly_performance_metrics[dcl.post_delivery_service_demands_duration_str][scenario-1][month-1]=np.sum(self.service_demands_durations_this_month[dcl.post_delivery_service_type_str])
        dcl.reporting_monthly_performance_metrics[dcl.total_service_demands_duration_str][scenario-1][month-1]=np.sum(self.service_demands_durations_this_month[dcl.pre_delivery_service_type_str])+np.sum(self.service_demands_durations_this_month[dcl.post_delivery_service_type_str])
        if month!=1:
            dcl.reporting_monthly_performance_metrics[dcl.backlog_service_demand_durations_str][scenario-1][month-1]=np.sum(self.service_demands_durations_backlog[(month-1,dcl.pre_delivery_service_type_str)]+self.service_demands_durations_backlog[(month-1,dcl.post_delivery_service_type_str)]) #backlog service of month 1 would be counted in month 2 #todo check


        return dcl.reporting_monthly_performance_metrics
    def comulative_summary_metrics(self,month,scenario):
        dcl.reporting_comulative_performance_metrics[dcl.number_of_services_str][scenario-1][month-1] = np.sum(dcl.reporting_monthly_performance_metrics[dcl.number_of_services_str][scenario-1][:month])
        dcl.reporting_comulative_performance_metrics[dcl.number_of_pre_delivery_services_str][scenario-1][month-1] = np.sum(dcl.reporting_monthly_performance_metrics[dcl.number_of_pre_delivery_services_str][scenario-1][:month])
        dcl.reporting_comulative_performance_metrics[dcl.number_of_post_delivery_services_str][scenario-1][month-1] = np.sum(dcl.reporting_monthly_performance_metrics[dcl.number_of_post_delivery_services_str][scenario-1][:month])


        dcl.reporting_comulative_performance_metrics[dcl.number_of_fulfilled_services_str][scenario-1][month-1] = np.sum(dcl.reporting_monthly_performance_metrics[dcl.number_of_fulfilled_services_str][scenario-1][:month])
        dcl.reporting_comulative_performance_metrics[dcl.number_of_fulfilled_pre_delivery_services_str][scenario-1][month-1] = np.sum(dcl.reporting_monthly_performance_metrics[dcl.number_of_fulfilled_pre_delivery_services_str][scenario-1][:month])
        dcl.reporting_comulative_performance_metrics[dcl.number_of_fulfilled_post_delivery_services_str][scenario-1][month-1]= np.sum(dcl.reporting_monthly_performance_metrics[dcl.number_of_fulfilled_post_delivery_services_str][scenario-1][:month])

        dcl.reporting_comulative_performance_metrics[dcl.technicians_worked_hours_str][scenario-1][month-1]=np.sum(dcl.reporting_monthly_performance_metrics[dcl.technicians_worked_hours_str][scenario-1][:month])
        return dcl.reporting_comulative_performance_metrics

    def all_attributes(self,month):
        attributes = {
        "service center id": self.id,
        "technicians":{'free':len([1 for technician in self.technicians if technician.is_available]), 'busy':len([1 for technician in self.technicians if not technician.is_available]) , 'total':len(self.technicians)},
        "vehicles in the queue for":{dcl.pre_delivery_service_type_str: len([1 for vehicle in self.queue if vehicle.service_type==dcl.pre_delivery_service_type_str]),dcl.post_delivery_service_type_str: len([1 for vehicle in self.queue if vehicle.service_type == dcl.post_delivery_service_type_str]),"total": len(self.queue) },
        "serviced vehicles":{dcl.pre_delivery_service_type_str:len( self.vehicles_serviced[(month,dcl.pre_delivery_service_type_str)]),dcl.post_delivery_service_type_str:len(self.vehicles_serviced[(month,dcl.post_delivery_service_type_str)]),'total':len(self.vehicles_serviced[(month,dcl.pre_delivery_service_type_str)]+self.vehicles_serviced[(month,dcl.post_delivery_service_type_str)])},
        "under service vehicles":{dcl.pre_delivery_service_type_str:len([1 for vehicle in self.vehicles_under_service if vehicle.service_type==dcl.pre_delivery_service_type_str]),dcl.post_delivery_service_type_str:len([1 for vehicle in self.vehicles_under_service if vehicle.service_type==dcl.post_delivery_service_type_str]),'total':len(self.vehicles_under_service)},

        }
        return attributes

#This function creates service centers objects
def create_service_centers(number_of_service_centers, max_working_hours_monthly,pre_delivery_service_duration_mean,pre_delivery_service_duration_variance,post_delivery_arrival_mean,post_delivery_arrival_variance,post_delivery_service_durations_events_data):
    service_centers = []

    for i in range(number_of_service_centers):
        log_info(f'create service center {i + 1}')
        service_centers.append(ServiceCenter(id=i + 1, max_working_hours=max_working_hours_monthly,pre_delivery_service_duration_mean=pre_delivery_service_duration_mean,pre_delivery_service_duration_variance=pre_delivery_service_duration_variance,post_delivery_arrival_mean=post_delivery_arrival_mean,post_delivery_arrival_variance=post_delivery_arrival_variance,post_delivery_service_durations_events_data=post_delivery_service_durations_events_data))
    return service_centers
def simulate_service_centers_performance(service_center,year,number_of_months,new_vehicles_monthly_volumes,number_of_technicians_to_hire,number_of_scenarios):

    #==========================================================================================
    print(dcl.section)
    log_info(f'simulating the operations of service center {service_center.id} in {year}')
    print(dcl.section)
    #=======================================================================
    #^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    #assumption:
    # 1. no vehicles are serviced by this service center before
    # 2. if a vehicle is serviced by this service center, it would always be serviced by the same service center
    #=======================================================================
    #add technicians to the service center
    for i in range(number_of_technicians_to_hire):
        service_center.add_technician(Technician(i+1))
    #^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    # assumption:
    # number of hired technicians remain the same over the year
    # can drop this assumption by hiring full-time and part-time technicians
    # number of full time technicians can be defined based on min of 95% confidence interval of the average service time
    # number of part-time technicians can be determined based on other service duration distribution statistics such as Q3 to manage the peak loads
    #^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    #========================================================================
    for scenario in range(1,number_of_scenarios+1):
        service_center_vehicles_volume = []  # list of vehicle volumes
        service_center.reset_for_new_scenarios()
        for month in range(1,number_of_months+1):

            vehicles_requested_for_post_delivery_service=[] #list of vehicles request for post-delivery services
            service_center.reset_monthly_service_demand_durations_list() #reset a list that records each month's service demand durations (backlog of previous months, unfulfilled services from previous months, are not included in this list)


            log_info(f"Date({month}-{year}) | upcoming pre-delivery service requests | {new_vehicles_monthly_volumes[month-1]}")
            log_info('generate new vehicles based on the vehicle demand volume') # Volume is assumed to be known and deterministic

            # new vehicles arrive at the service center to get pre-delivery services
            #these vehicles are added to the service center's queue
            #queueing system : first-come first-serve
            # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
            #assumptions:
            # All service demand request are revealed at the beginning of each month ; they arrive at the hour=0 of each month
            # In the first month, only pre-delivery services are requested; i.e. no vehicles are on the road to request for post delivery services
            # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
            vehicles_requested_for_pre_delivery_service=service_center.generate_new_vehicles(month=month,pre_delivery_volume=new_vehicles_monthly_volumes[month-1])

            #adding the new produced vehicles to the list of
            service_center_vehicles_volume.extend(vehicles_requested_for_pre_delivery_service)
            vehicles_on_the_road_not_requested_service = [vehicle for vehicle in service_center_vehicles_volume if vehicle.service_type not in ['pre-delivery', 'post-delivery']]
            if len(vehicles_on_the_road_not_requested_service)!=0: #if some vehicles are in the service center territory
                #count how many of them 1. finished it's predelivery service and 2. have not yet requested for any post-delievry services services. these vehicles are potential candidats for post-delivery service this month
                log_info(f'{len(vehicles_on_the_road_not_requested_service)} vehicles are on the road and may request for post-delivery service')
                log_info('generate post delivery requests')

                #generate post delivery service requests using 1. number of vehicles on the road, 2. post-delivery arrival mean, 3.post-delivery arrival variance
                #these vehicles are added to the service center's queue
                vehicles_requested_for_post_delivery_service=service_center.generate_post_delivery_requests(month=month,vehicles_on_the_road=vehicles_on_the_road_not_requested_service)
                log_info(f"Date({month}-{year}) | upcoming post-delivery service requests | {len(vehicles_requested_for_post_delivery_service)}")
            #=========================================================================================================================
            # service centers status at the beginning of the month
            #
            print('-'*120)
            print(f"{'':<50}status info: service center {'':<50}")
            print(service_center.all_attributes(month))
            log_objects_status(f"Technicians status ",service_center.technicians)
            log_objects_status(f"Queue status ",service_center.queue)

            #==============================================================================================================================
            # Simulate the month
            log_info(f" simulation started")
            service_center.simulate_month(month)
            log_info(info_str=f"simulation finished")
            log_objects_status(info_str=f"({dcl.max_working_hours_service_center_in_month}-{month}-{dcl.year})|",val=service_center)
           #-------------------------------------------------
            # updating the statuses for the next round of simulation

            for vehicle in service_center.queue:
                vehicle.waiting_time += service_center.max_working_hours
                service_center.queue_waiting_time[month].append(vehicle.waiting_time)
                service_center.service_demands_durations_backlog[(month, vehicle.service_type)].append(vehicle.service_duration)
            for vehicle in service_center.vehicles_under_service:
                service_center.service_demands_durations_backlog[(month, vehicle.service_type)].append(vehicle.remaining_service_time)

            monthly_performance_report=service_center.monthly_summary_metrics(month=month,scenario=scenario,new_generated_vehicles_this_month=vehicles_requested_for_pre_delivery_service,vehicles_post_delivery_service_this_month=vehicles_requested_for_post_delivery_service)
            cumulative_performance_report=service_center.comulative_summary_metrics(month=month,scenario=scenario)


            for technician in service_center.technicians:
                technician.reset_new_month()
            if not dcl.disable_print_summary_metrics:
                log_summary_metrics(month=month,monthly_summary_metrics=monthly_performance_report,comulative_summary_metrics=cumulative_performance_report)
            log_info(info_str='rolling forward to simulate the next month')
            print(section)
    return monthly_performance_report,cumulative_performance_report


def plot_monthly_performance_metric(df, service_center_id, performance_metric):
    """
    Plots the specified performance metric across months for a given service center.

    Parameters:
    - df: DataFrame containing the performance metrics, service center IDs, number of technicians, and months.
    - service_center_id: The ID of the service center to plot the metrics for.
    - performance_metric: The name of the performance metric to plot.
    """
    # Filter the DataFrame for the specified service center ID
    df_filtered = df[df['Service Center ID'] == service_center_id]

    # Ensure data is sorted by month and then by number of technicians
    df_filtered.sort_values(by=['Month', 'Number of Technicians'], inplace=True)

    # Plotting
    plt.figure(figsize=(10, 6))

    # Unique number of technicians
    technicians_list = df_filtered['Number of Technicians'].unique()

    for technician in technicians_list:
        # Filter the DataFrame for each number of technicians
        df_technician = df_filtered[df_filtered['Number of Technicians'] == technician]

        plt.plot(df_technician['Month'], df_technician[performance_metric], marker='o',
                 label=f'{technician} Technicians')

    plt.title(f'Service Center {service_center_id} - {performance_metric} by Month')
    plt.xlabel('Month')
    plt.ylabel(performance_metric)
    plt.xticks(df_filtered['Month'].unique())  # Ensure x-ticks are the months
    plt.legend()
    plt.grid(True)
    plt.show()
