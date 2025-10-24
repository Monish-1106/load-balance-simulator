# Load Balancing in Parallel Computing

## Project Overview

This project simulates **load balancing** in a parallel computing system, distributing tasks among multiple processors using **Round-Robin** and **Best-Fit-Decreasing** algorithms. It analyzes processor workloads, makespan, utilization, and imbalance to help identify performance bottlenecks and improve efficiency.

## Features

- Generates a batch of tasks with different workload distributions:
  - Uniform
  - Exponential
  - Pareto
- Supports **homogeneous or heterogeneous processors** with different speeds.
- Implements two scheduling algorithms:
  - **Round-Robin**: assigns tasks in cyclic order.
  - **Best-Fit-Decreasing**: assigns tasks to the processor with minimum current load.
- Calculates performance metrics:
  - Makespan
  - Total work time
  - Average utilization
  - Load imbalance
- Visualizes per-processor loads with **Matplotlib** bar charts.

## Requirements

- Python 3.8+
- Libraries:
  ```bash
  pip install numpy matplotlib

## How to Run

Open the terminal in the project folder.

(Optional) Activate your virtual environment:

powershell
Copy code
.\.venv\Scripts\activate
Run the simulation:

powershell
Copy code
python load_balance_sim.py
Metrics will print in the terminal, and a bar chart will pop up comparing Round-Robin vs Best-Fit-Decreasing.

## File Structure

bash
Copy code
COA_PROJECT/
├── load_balance_sim.py      # Main Python script
├── lb_result.png            # Saved chart (generated after running)
├── README.md                # Project documentation
└── .venv/                   # Virtual environment (optional)

## Author

Name: R.R Monish Raj
Course: B.Tech CSE – AI/ML
Institute: SRM Institute of Science and Technology

## License

This project is created for educational purposes.
You are free to modify, reuse, and extend it for academic demonstrations.


This project is created for educational purposes.
You are free to modify, reuse, and extend it for academic demonstrations.
