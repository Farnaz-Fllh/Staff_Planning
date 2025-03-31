# StaffPlanalytics: A Service Center Simulation Toolkit

Welcome to `StaffPlanalytics`, a Python-based simulation framework designed to help optimize staffing decisions in service center environments. This repository provides both a strategic and technical toolkit for workforce planning using discrete event simulation.

---

## 🚀 Project Purpose

This simulation project aims to:
- Model real-world service center operations
- Test technician staffing scenarios
- Visualize queue lengths, service durations, and utilization
- Provide actionable insights into bottlenecks and customer satisfaction trade-offs

---

## 🔧 Key Features

- **Discrete Event Simulation (DES)**
- **Object-Oriented Design (Vehicle, Technician, ServiceCenter)**
- **First-Come-First-Serve Queue Logic**
- **Scenario Analysis (e.g., changing staffing levels, demand distributions, service times, etc)**
- **Agent-based modeling with monthly time steps**
- **Provide Insights on Important Business KPIs under Various Scenarios**

---

## 📂 Repository Contents

```bash
StaffPlanalytics/
├── declare.py              # Simulation inputs and config
├── utils.py                # Simulation logic and core classes
├── main.py                 # Execution script
├── README.md               # Project overview (this file)
├── USER_MANUAL.md          # Full technical documentation
├── Report.pdf / .ipynb     # Final analysis report
└── all_service_centers_average_monthly_performance_metrics.csv
```

---

## 🧪 How to Run the Simulation

```bash
# Install required packages
pip install -r requirements.txt

# Execute main simulation
python main.py
```

Results will be saved as `all_service_centers_average_monthly_performance_metrics.csv`, summarizing key KPIs for each simulated configuration.

---

## 📊 Example Metrics Tracked

- Number of services fulfilled (monthly & cumulative)
- Technician utilization
- Queue wait times
- Backlog duration
- Total service time requested vs. fulfilled

---

## 📘 Documentation

For detailed class structures, assumptions, and code explanations, please refer to the [User Manual](User_Manual.md).

---

## 🧠 Why This Project Matters

Workforce planning in service operations is complex. This toolkit helps planners:
- Visualize the effect of technician count on KPIs
- Analyze bottlenecks before scaling
- Make data-driven staffing decisions

---

## 📄 License

This project is licensed under the MIT License.

---

## 🤝 Connect
If you're interested in collaborating or want to learn more about the modeling techniques behind this toolkit, feel free to reach out!

