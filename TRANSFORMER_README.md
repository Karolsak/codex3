# Advanced Transformer Efficiency Calculator & Dynamic Simulator

A comprehensive Python application for electrical engineering calculations and real-time transformer simulations.

## Problem Solution

**Given:**
- Single-phase transformer working at unity power factor
- Efficiency = 90% at both half-load and full-load of 1 kW
- **Find:** Efficiency at 70% of full-load

**Solution:**

The transformer has:
- **Core Loss:** 66.67 W (constant)
- **Copper Loss (Full Load):** 44.44 W (varies with load²)
- **Efficiency at 70% load:** 89.49%

### Mathematical Approach:

1. At full load (1000W): η = P_out / (P_out + P_core + P_cu) = 0.90
2. At half load (500W): η = P_out / (P_out + P_core + 0.25*P_cu) = 0.90
3. Solving simultaneously: P_core = 66.67W, P_cu = 44.44W
4. At 70% load: η = 700 / (700 + 66.67 + 0.49*44.44) = 89.49%

## Features

### 1. Efficiency Calculator Tab
- **Input Parameters:** Configure full load, full-load efficiency, and half-load efficiency
- **Real-time Calculation:** Automatically calculates core and copper losses
- **Interactive Slider:** Adjust load from 10% to 150% and see efficiency changes
- **Visualization:** Dynamic efficiency vs load curve with marked optimal points
- **Maximum Efficiency Point:** Automatically identifies the load for peak efficiency

### 2. Dynamic Simulation Tab
- **Real-time ODE Solver:** Choose between RK45 (Runge-Kutta 4th/5th order), RK23, or Euler method
- **Interactive Controls:**
  - Input voltage slider (0-400V)
  - Load resistance slider (10-200Ω)
  - Simulation time configuration
- **Differential Equations:** Models transformer behavior with:
  - Primary and secondary currents
  - Magnetic flux linkage
  - Voltage-current relationships
- **Three Real-time Plots:**
  - Primary and secondary currents
  - Magnetic flux
  - Input and output power
- **Control Buttons:** Start, Stop, Reset simulation

### 3. Advanced Analysis Tab
- **Loss Analysis:**
  - Core, copper, and total losses vs load
  - Efficiency curve
  - Loss distribution pie chart
  - Power flow visualization

- **Optimization Analysis:**
  - All-day efficiency for different load profiles (Industrial, Commercial, Residential)
  - Optimal operating region identification
  - Annual cost of losses calculation
  - Voltage regulation analysis

- **24-Hour Load Profile:**
  - Simulated daily load patterns
  - Real-time efficiency tracking
  - Cumulative energy output and losses
  - Daily summary statistics

## Technical Implementation

### Mathematical Models

**1. Efficiency Calculation:**
```
η(load) = P_out / (P_out + P_core + P_cu * load²)
```

**2. Transformer Differential Equations:**
```
V_in = R_p*i_p + L_p*di_p/dt + M*di_s/dt
0 = R_s*i_s + L_s*di_s/dt + M*di_p/dt + R_load*i_s
dφ/dt = M * i_p
```

Where:
- i_p, i_s: Primary and secondary currents
- φ: Magnetic flux
- L_p, L_s: Primary and secondary inductances
- M: Mutual inductance
- R_p, R_s: Winding resistances

### ODE Solvers

**1. RK45 (Default):**
- Adaptive step-size Runge-Kutta method
- 4th and 5th order accurate
- Excellent for stiff problems
- Best accuracy-to-speed ratio

**2. RK23:**
- Runge-Kutta 2nd/3rd order
- Faster than RK45 for simple problems
- Good for non-stiff equations

**3. Euler Method:**
- First-order explicit method
- Simple and fast
- Educational purposes
- Fixed time step (dt = 0.001s)

## Installation

### Requirements:
```bash
pip install numpy matplotlib scipy tkinter
```

Note: `tkinter` usually comes pre-installed with Python.

## Usage

### Run the Application:
```bash
python3 transformer_efficiency_calculator.py
```

### Quick Start:

1. **Efficiency Calculator:**
   - Enter full load power, full-load efficiency, and half-load efficiency
   - Click "Calculate Losses"
   - Use the slider to explore efficiency at different loads

2. **Dynamic Simulation:**
   - Adjust input voltage and load resistance
   - Select solver from the menu (Solver → RK45/Euler/RK23)
   - Click "Start" to run simulation
   - Watch real-time waveforms

3. **Advanced Analysis:**
   - Click "Generate Loss Analysis" for comprehensive loss breakdown
   - Click "Optimization Analysis" for efficiency optimization insights
   - Click "Load Profile Analysis" for 24-hour simulation

## Key Features for Electrical Engineering

### Practical Applications:

1. **Transformer Sizing:** Determine optimal transformer rating for given load profiles
2. **Loss Calculation:** Estimate energy losses and operating costs
3. **Efficiency Optimization:** Identify best operating points
4. **Load Planning:** Analyze daily/seasonal load variations
5. **Cost Analysis:** Calculate annual energy loss costs
6. **Design Verification:** Validate transformer designs with dynamic simulations

### Advanced Capabilities:

- **Auto-scaling Plots:** Automatically adjust to window size
- **Professional UI:** Clean, intuitive interface with tabbed design
- **Multi-threaded Simulation:** Non-blocking UI during calculations
- **Comprehensive Visualization:** Multiple plot types (line, bar, pie, filled)
- **Real-time Updates:** Live efficiency calculations as parameters change
- **Export-ready Plots:** High-resolution matplotlib figures

## Technical Specifications

- **Language:** Python 3.x
- **GUI Framework:** Tkinter
- **Plotting:** Matplotlib with TkAgg backend
- **Numerical Methods:** NumPy, SciPy
- **ODE Solvers:** Custom Euler implementation + SciPy solve_ivp
- **Resolution:** Adaptive window sizing with automatic scaling

## Menu Options

### File Menu:
- **Reset All:** Clear all parameters and plots
- **Exit:** Close application

### Solver Menu:
- **RK45 (Adaptive):** High-accuracy adaptive solver
- **Euler Method:** Simple first-order method
- **RK23:** Medium-accuracy solver

### Help Menu:
- **About:** Application information

## Example Results

For the given problem:
- **Core Loss:** 66.67 W
- **Copper Loss (Full):** 44.44 W
- **Efficiency at 70% load:** 89.49%
- **Maximum Efficiency Load:** 122.5%
- **Maximum Efficiency:** 90.00%

## Error Handling

- Input validation for all numerical entries
- Graceful handling of invalid parameters
- Thread-safe simulation control
- Automatic range limiting for sliders

## Future Enhancements

- Three-phase transformer support
- Temperature effects on resistance
- Harmonic analysis
- Load flow calculations
- Export to CSV/Excel
- Custom load profile import
- Comparative analysis tools

## Author Notes

This application combines theoretical transformer analysis with practical engineering tools. It's designed for:
- Electrical engineering students
- Power system engineers
- Transformer designers
- Energy auditors
- Academic research

The dynamic simulation uses authentic differential equations to model transformer behavior, making it suitable for both educational and professional use.

## License

Free for educational and non-commercial use.

---

**Note:** All calculations assume ideal sinusoidal waveforms and unity power factor unless otherwise specified. Real-world transformers may have additional factors affecting performance.
