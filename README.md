# Rivian_Staff_Planning_Assignment


# Rivian Service Planning Solution

Welcome to my GitHub repository, where I've compiled my analysis, thought process, and recommendations for addressing the service planning challenges faced by Rivian. This work aims to provide actionable insights and tools to enhance the efficiency and effectiveness of Rivian's service center operations.

## Final Report

The repository includes the final report in two formats for convenience:
- **PDF Format**: `Report.pdf`
- **Jupyter Notebook Format**: `Report.ipynb`

These documents detail the analytical process, findings, and strategic recommendations developed through comprehensive analysis.

## StaffPlanalytics: A Simulation Package

As a pivotal part of the solution, I introduce `StaffPlanalytics`, an object-oriented simulation package developed in Python. This package is designed to empower the Rivian team to assess and optimize the performance of service centers under various staffing scenarios.

---

**Introducing `StaffPlanalytics` for Optimizing Technician Staffing Levels**

---

### Overview

`StaffPlanalytics` is a "Python package" designed to assist staff planning of service centers. The package simulates the operations of service centers based on given number of technicians, detailed service duration data and service demands arrival patterns. This package leverages discrete event simulation and agenet-based modeling techniques to provide actionable insights for service center management. It mimics the real-world service center operations and enables visualizing the impact of different staffing levels on service efficiency and customer wait times.

The developed DES package provides the following advantages: 

1. **Realistic Operational Modeling**: modeling detailed operational process of service centers, including technician schedules, service duration variability, customer arrival patterns, and queue management rules. 

2. **Queue Management Analysis**: Simulating service requests arrivals, service durations, service completions to track the queue lengths and wait times under different staffing levels as well as service duration and volume scenarios for each month. This helps identify whether the current number of technicians can manage peak demand periods without causing excessive delays. The implemented queue management logic is **first come, first serve**. 

3. **Resource Utilization**: Tracking the utilization rates of technicians in terms of number of services finished and the duration of services. This allows us to identify periods of underutilization (suggesting overstaffing) or constant high utilization (indicating potential understaffing or the need for process improvements).

4. **Scenario Testing**: Allowing for "what-if" analyses under various conditions, such as increase in service demand volumes for particular service types, changes in service duration distributions, or variations in technician efficiency (introducing technicians fatigue or skill level factors). This helps in determining the robustness of current staffing levels against possible future changes.

5. **Bottleneck Identification**: The developed DES allows us to analyze the flow of services through service centers. It allows us to identify bottlenecks where services might be delayed, indicating areas where additional technicians or process improvements could enhance overall efficiency.

7. **Customer Satisfaction Metrics**: Beyond just measuring service times and utilization, the package allows us to simulate the impact of staffing levels on customer satisfaction metrics, such as the percentage of services completed within a target timeframe. We specifically track the number of fulfilled services each month by service type and in total. 

### Getting Started

The `StaffPlanalytics` consists of three modules: `declare`, `utils`, and `main`. The declare modules defines all input parameters and required data for the simulation. The `utils` module contains all functions required to run the simulation. The `main` module runs the simulation process. For more information, please refer to the StaffPlanalytics user manual file.


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
