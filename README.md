# Staff Planning Simulation Toolkit

Welcome to this repository showcasing an end-to-end simulation-based approach to solving staff planning challenges in service center operations. This work presents a comprehensive solution for analyzing technician utilization, customer wait times, and operational efficiency under various staffing scenarios.

## Final Report

The repository includes the final report in two formats for convenience:
- **PDF Format**: [Report.pdf](Report.pdf)
- **Jupyter Notebook Format**: [Report.ipynb](Report.ipynb)

These documents outline the analytical process, findings, and strategic recommendations based on simulation modeling.

## StaffPlanalytics: A Simulation Package

As a core component of the solution, this project includes `StaffPlanalytics`, an object-oriented Python package developed to assess and optimize service center operations through discrete event simulation (DES).

---

**Introducing `StaffPlanalytics` for Optimizing Technician Staffing Levels**

---

### Overview

`StaffPlanalytics` is a modular simulation package designed to support service center workforce planning. It simulates technician workloads, service demand arrival patterns, and queue management dynamics to evaluate the performance of different staffing configurations.

This tool leverages discrete event simulation and agent-based modeling to mimic real-world operations and quantify the impact of technician staffing levels on customer service outcomes.

### Key Features
- **Object-Oriented Design**: Enables reusability, easy integration, and future scalability.
- **Modular Architecture**: Organized into three key components — `main`, `declare`, and `utils` — each responsible for distinct tasks in the simulation workflow.
- **Performance Analysis**: Supports in-depth examination of operational bottlenecks, queue behaviors, and technician utilization.

### Modules Overview
- **Main Module**: Orchestrates the simulation run and integrates other components.
- **Declare Module**: Reads input data and defines simulation parameters, such as historical service volumes, service durations, technician counts, and service types.
- **Utils Module**: Contains simulation entities (technicians, vehicles, service centers) and utility functions for data processing, simulation mechanics, and visualization.

### Key Capabilities

1. **Realistic Operational Modeling**: Incorporates service variability, technician schedules, and real-world queue dynamics.
2. **Queue Management Analysis**: Simulates first-come-first-serve logic to measure queue lengths and customer wait times across multiple scenarios.
3. **Resource Utilization Tracking**: Quantifies technician productivity and helps identify over/understaffed periods.
4. **Scenario Testing**: Enables “what-if” analysis to evaluate system resilience under changes in demand, service time, or technician performance.
5. **Bottleneck Identification**: Detects critical constraints in service flow and suggests areas for process improvements.
6. **Customer Satisfaction Metrics**: Tracks fulfillment rates, service time thresholds, and other KPIs to assess the customer impact of staffing decisions.

## Simulation Results: Performance Metrics

The file [`all_service_centers_average_monthly_performance_metrics.csv`](all_service_centers_average_monthly_performance_metrics.csv) contains summary metrics across multiple staffing configurations (1 to 5 technicians per center) for two service centers, based on simulations run for the year 2023. These metrics provide insights into optimal staffing strategies based on wait times, service completion rates, and technician utilization.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
