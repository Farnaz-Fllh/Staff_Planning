# StaffPlanalytics User Manual

This document provides detailed technical guidance on using the `StaffPlanalytics` simulation package for service center workforce planning. It is intended for developers, analysts, or operational planners seeking to understand and extend the functionality.

---

## 📦 Overview

`StaffPlanalytics` is a modular Python package that uses discrete event simulation (DES) and agent-based modeling (ABM) to replicate service center operations and analyze technician staffing scenarios. It helps identify bottlenecks, optimize technician levels, and improve service center performance.

## 🧠 Core Capabilities

1. **Realistic Operational Modeling**
2. **Queue Management (First Come, First Serve)**
3. **Technician Utilization Tracking**
4. **Scenario Testing (what-if analysis)**
5. **Bottleneck Identification**
6. **Customer Satisfaction Metrics**

---

## ⚙️ Simulation Architecture

The package consists of three modules:

- `declare.py`: Defines input parameters and simulation constants.
- `utils.py`: Houses core classes (Vehicle, Technician, ServiceCenter) and utility functions.
- `main.py`: Runs the simulation by orchestrating the flow.

### Class Summary
The package defines three main classes to simulate interactions:
- `Vehicle`
- `Technician`
- `ServiceCenter`

---

## ▶️ Running a Simulation
The `simulate_service_centers_performance()` function runs month-by-month simulations over multiple scenarios. It models vehicle arrivals, technician schedules, queue processing, service completions, and performance metrics.
Example flow from `main.py`:

```python
from utils import *
import declare as dcl

# Step 1: Calculate monthly volume
_, service_center_monthly_volumes_dict = calculate_service_centers_monthly_delivered_volume(dcl.service_center_delivered_df)

# Step 2: Create service center objects
service_centers = create_service_centers(...)

# Step 3: Run simulation
for service_center in service_centers:
    for number_of_technicians in range(dcl.minimum_number_of_technicians, dcl.maximum_number_of_technicians + 1):
        monthly_performance_report, _ = simulate_service_centers_performance(...)
        ...
        all_monthly_performance_metrics.append(df_current_monthly)

# Export results
pd.concat(all_monthly_performance_metrics).to_csv('all_service_centers_average_monthly_performance_metrics.csv')
```

---

## 📊 Output

Main output: `all_service_centers_average_monthly_performance_metrics.csv`

Each row contains:
- Service center ID
- Month
- Number of technicians
- KPIs: queue length, waiting time, utilization, number of services completed, etc.

---

## 🔍 Performance Metrics
- Number and type of services requested
- Fulfilled vs. backlog services
- Queue waiting time
- Technician utilization rates

---

## 📁 Folder Structure

```bash
StaffPlanalytics/
├── declare.py              # Inputs and configs
├── utils.py                # Classes and helpers
├── main.py                 # Simulation script
├── USER_MANUAL.md          # This file
├── README.md               # Project overview
├── Report.pdf / .ipynb     # Final analysis
└── all_service_centers_average_monthly_performance_metrics.csv

---

## 📄 License
This project is licensed under the MIT License.
