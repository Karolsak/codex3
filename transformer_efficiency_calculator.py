"""
Advanced Transformer Efficiency Calculator and Dynamic Simulator
A comprehensive tool for electrical engineering calculations and real-time simulations
"""

import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import threading
import time
from scipy.integrate import odeint, solve_ivp


class TransformerModel:
    """Mathematical model for transformer behavior"""

    def __init__(self):
        self.core_loss = 0.0
        self.copper_loss_full = 0.0
        self.full_load = 1000  # Watts
        self.reset_parameters()

    def reset_parameters(self):
        """Reset all parameters to default values"""
        self.core_loss = 0.0
        self.copper_loss_full = 0.0
        self.full_load = 1000

    def calculate_losses_from_efficiency(self, eff_full, eff_half, full_load_watts):
        """
        Calculate core and copper losses from given efficiency data

        Args:
            eff_full: Efficiency at full load (0-1)
            eff_half: Efficiency at half load (0-1)
            full_load_watts: Full load power in watts

        Returns:
            tuple: (core_loss, copper_loss_full)
        """
        self.full_load = full_load_watts

        # At full load: η_fl = P_out / (P_out + P_core + P_cu)
        # Rearranging: P_core + P_cu = P_out * (1/η_fl - 1)
        total_loss_full = full_load_watts * (1/eff_full - 1)

        # At half load: copper losses are (0.5)^2 = 0.25 of full load
        half_load = full_load_watts / 2
        total_loss_half = half_load * (1/eff_half - 1)

        # P_core + 0.25*P_cu = total_loss_half
        # P_core + P_cu = total_loss_full
        # Solving: P_cu = (total_loss_full - total_loss_half) / 0.75
        self.copper_loss_full = (total_loss_full - total_loss_half) / 0.75
        self.core_loss = total_loss_full - self.copper_loss_full

        return self.core_loss, self.copper_loss_full

    def calculate_efficiency(self, load_fraction):
        """
        Calculate efficiency at given load fraction

        Args:
            load_fraction: Load as fraction of full load (0-1)

        Returns:
            float: Efficiency (0-1)
        """
        output_power = self.full_load * load_fraction
        copper_loss = self.copper_loss_full * (load_fraction ** 2)
        total_loss = self.core_loss + copper_loss

        if output_power + total_loss == 0:
            return 0

        efficiency = output_power / (output_power + total_loss)
        return efficiency

    def calculate_max_efficiency_load(self):
        """Calculate load fraction for maximum efficiency"""
        if self.copper_loss_full == 0:
            return 0
        # Maximum efficiency occurs when core loss = copper loss
        # P_core = P_cu * load^2
        # load = sqrt(P_core / P_cu)
        max_eff_load = np.sqrt(self.core_loss / self.copper_loss_full)
        return max_eff_load

    def transformer_dynamics(self, t, y, V_in, R_load):
        """
        Differential equations for transformer dynamics

        State variables:
        y[0] = primary current (i_p)
        y[1] = secondary current (i_s)
        y[2] = magnetic flux (φ)

        Args:
            t: time
            y: state vector [i_p, i_s, phi]
            V_in: input voltage
            R_load: load resistance

        Returns:
            derivatives: [di_p/dt, di_s/dt, dphi/dt]
        """
        i_p, i_s, phi = y

        # Transformer parameters
        L_p = 0.1  # Primary inductance (H)
        L_s = 0.1  # Secondary inductance (H)
        M = 0.09   # Mutual inductance (H)
        R_p = 0.5  # Primary resistance (Ω)
        R_s = 0.5  # Secondary resistance (Ω)

        # Voltage equations
        # V_in = R_p*i_p + L_p*di_p/dt + M*di_s/dt
        # 0 = R_s*i_s + L_s*di_s/dt + M*di_p/dt + R_load*i_s

        # Rearranging for derivatives
        det = L_p * L_s - M * M

        di_p_dt = (L_s * (V_in - R_p * i_p) - M * (-R_s * i_s - R_load * i_s)) / det
        di_s_dt = (M * (-V_in + R_p * i_p) + L_p * (-R_s * i_s - R_load * i_s)) / det
        dphi_dt = M * i_p

        return [di_p_dt, di_s_dt, dphi_dt]

    def euler_method(self, func, y0, t_span, args=()):
        """
        Euler method for solving ODEs

        Args:
            func: derivative function
            y0: initial conditions
            t_span: [t_start, t_end]
            args: additional arguments for func

        Returns:
            tuple: (t_array, y_array)
        """
        dt = 0.001  # Time step
        t_start, t_end = t_span
        t = np.arange(t_start, t_end, dt)
        y = np.zeros((len(t), len(y0)))
        y[0] = y0

        for i in range(1, len(t)):
            dydt = func(t[i-1], y[i-1], *args)
            y[i] = y[i-1] + np.array(dydt) * dt

        return t, y


class TransformerSimulator:
    """Main application class"""

    def __init__(self, root):
        self.root = root
        self.root.title("Advanced Transformer Efficiency Calculator & Simulator")
        self.root.geometry("1400x900")

        # Configure grid weight for resizing
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)

        # Model and simulation variables
        self.model = TransformerModel()
        self.simulation_running = False
        self.simulation_thread = None
        self.solver_method = "RK45"  # Default solver

        # Data storage for plotting
        self.time_data = []
        self.efficiency_data = []
        self.load_data = []
        self.loss_data = []
        self.current_time = 0

        # Create main container
        self.main_container = ttk.Frame(root)
        self.main_container.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        self.main_container.grid_rowconfigure(1, weight=1)
        self.main_container.grid_columnconfigure(0, weight=1)

        # Create UI components
        self.create_menu()
        self.create_notebook()

        # Bind resize event
        self.root.bind('<Configure>', self.on_resize)

        # Initialize with default calculation
        self.solve_example_problem()

    def create_menu(self):
        """Create menu bar"""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)

        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Reset All", command=self.reset_all)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)

        # Solver menu
        solver_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Solver", menu=solver_menu)
        solver_menu.add_command(label="RK45 (Adaptive)", command=lambda: self.set_solver("RK45"))
        solver_menu.add_command(label="Euler Method", command=lambda: self.set_solver("Euler"))
        solver_menu.add_command(label="RK23", command=lambda: self.set_solver("RK23"))

        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="About", command=self.show_about)

    def create_notebook(self):
        """Create tabbed interface"""
        self.notebook = ttk.Notebook(self.main_container)
        self.notebook.grid(row=0, column=0, sticky="nsew", pady=5)

        # Create tabs
        self.tab_efficiency = ttk.Frame(self.notebook)
        self.tab_dynamic = ttk.Frame(self.notebook)
        self.tab_analysis = ttk.Frame(self.notebook)

        self.notebook.add(self.tab_efficiency, text="Efficiency Calculator")
        self.notebook.add(self.tab_dynamic, text="Dynamic Simulation")
        self.notebook.add(self.tab_analysis, text="Advanced Analysis")

        # Configure tab grids
        self.tab_efficiency.grid_rowconfigure(1, weight=1)
        self.tab_efficiency.grid_columnconfigure(0, weight=1)

        self.tab_dynamic.grid_rowconfigure(1, weight=1)
        self.tab_dynamic.grid_columnconfigure(0, weight=1)

        self.tab_analysis.grid_rowconfigure(0, weight=1)
        self.tab_analysis.grid_columnconfigure(0, weight=1)

        # Create content for each tab
        self.create_efficiency_tab()
        self.create_dynamic_tab()
        self.create_analysis_tab()

    def create_efficiency_tab(self):
        """Create efficiency calculator interface"""
        # Input frame
        input_frame = ttk.LabelFrame(self.tab_efficiency, text="Input Parameters", padding=10)
        input_frame.grid(row=0, column=0, sticky="ew", padx=5, pady=5)

        # Full load input
        ttk.Label(input_frame, text="Full Load Power (W):").grid(row=0, column=0, sticky="w", pady=5)
        self.full_load_var = tk.StringVar(value="1000")
        ttk.Entry(input_frame, textvariable=self.full_load_var, width=15).grid(row=0, column=1, padx=5, pady=5)

        # Full load efficiency
        ttk.Label(input_frame, text="Full Load Efficiency (%):").grid(row=1, column=0, sticky="w", pady=5)
        self.eff_full_var = tk.StringVar(value="90")
        ttk.Entry(input_frame, textvariable=self.eff_full_var, width=15).grid(row=1, column=1, padx=5, pady=5)

        # Half load efficiency
        ttk.Label(input_frame, text="Half Load Efficiency (%):").grid(row=2, column=0, sticky="w", pady=5)
        self.eff_half_var = tk.StringVar(value="90")
        ttk.Entry(input_frame, textvariable=self.eff_half_var, width=15).grid(row=2, column=1, padx=5, pady=5)

        # Calculate button
        ttk.Button(input_frame, text="Calculate Losses", command=self.calculate_losses).grid(
            row=3, column=0, columnspan=2, pady=10)

        # Results frame
        results_frame = ttk.LabelFrame(self.tab_efficiency, text="Calculated Parameters", padding=10)
        results_frame.grid(row=0, column=1, sticky="nsew", padx=5, pady=5, rowspan=1)

        self.core_loss_label = ttk.Label(results_frame, text="Core Loss: -- W", font=("Arial", 10, "bold"))
        self.core_loss_label.grid(row=0, column=0, sticky="w", pady=5)

        self.copper_loss_label = ttk.Label(results_frame, text="Copper Loss (Full): -- W", font=("Arial", 10, "bold"))
        self.copper_loss_label.grid(row=1, column=0, sticky="w", pady=5)

        self.max_eff_load_label = ttk.Label(results_frame, text="Max Efficiency Load: -- %", font=("Arial", 10, "bold"))
        self.max_eff_load_label.grid(row=2, column=0, sticky="w", pady=5)

        # Load slider frame
        slider_frame = ttk.LabelFrame(self.tab_efficiency, text="Load Control", padding=10)
        slider_frame.grid(row=0, column=2, sticky="nsew", padx=5, pady=5)
        slider_frame.grid_columnconfigure(0, weight=1)

        ttk.Label(slider_frame, text="Load Fraction:").grid(row=0, column=0, sticky="w", pady=5)
        self.load_var = tk.DoubleVar(value=0.7)
        self.load_slider = ttk.Scale(slider_frame, from_=0.1, to=1.5, orient="horizontal",
                                      variable=self.load_var, command=self.update_efficiency)
        self.load_slider.grid(row=1, column=0, sticky="ew", pady=5, padx=5)

        self.load_value_label = ttk.Label(slider_frame, text="70.0%", font=("Arial", 12, "bold"))
        self.load_value_label.grid(row=2, column=0, pady=5)

        self.efficiency_result_label = ttk.Label(slider_frame, text="Efficiency: -- %",
                                                  font=("Arial", 14, "bold"), foreground="blue")
        self.efficiency_result_label.grid(row=3, column=0, pady=10)

        # Visualization frame
        viz_frame = ttk.LabelFrame(self.tab_efficiency, text="Efficiency vs Load Curve", padding=5)
        viz_frame.grid(row=1, column=0, columnspan=3, sticky="nsew", padx=5, pady=5)
        viz_frame.grid_rowconfigure(0, weight=1)
        viz_frame.grid_columnconfigure(0, weight=1)

        # Create matplotlib figure
        self.fig_efficiency = Figure(figsize=(10, 4), dpi=100)
        self.ax_efficiency = self.fig_efficiency.add_subplot(111)
        self.canvas_efficiency = FigureCanvasTkAgg(self.fig_efficiency, viz_frame)
        self.canvas_efficiency.get_tk_widget().grid(row=0, column=0, sticky="nsew")

        # Configure column weights
        input_frame.grid_columnconfigure(1, weight=1)
        self.tab_efficiency.grid_columnconfigure(0, weight=1)
        self.tab_efficiency.grid_columnconfigure(1, weight=1)
        self.tab_efficiency.grid_columnconfigure(2, weight=1)

    def create_dynamic_tab(self):
        """Create dynamic simulation interface"""
        # Control frame
        control_frame = ttk.LabelFrame(self.tab_dynamic, text="Simulation Controls", padding=10)
        control_frame.grid(row=0, column=0, sticky="ew", padx=5, pady=5)

        # Voltage slider
        ttk.Label(control_frame, text="Input Voltage (V):").grid(row=0, column=0, sticky="w", pady=5)
        self.voltage_var = tk.DoubleVar(value=230)
        ttk.Scale(control_frame, from_=0, to=400, orient="horizontal",
                  variable=self.voltage_var, command=self.update_voltage_label).grid(row=0, column=1, sticky="ew", padx=5)
        self.voltage_label = ttk.Label(control_frame, text="230.0 V", width=10)
        self.voltage_label.grid(row=0, column=2, padx=5)

        # Load resistance slider
        ttk.Label(control_frame, text="Load Resistance (Ω):").grid(row=1, column=0, sticky="w", pady=5)
        self.resistance_var = tk.DoubleVar(value=50)
        ttk.Scale(control_frame, from_=10, to=200, orient="horizontal",
                  variable=self.resistance_var, command=self.update_resistance_label).grid(row=1, column=1, sticky="ew", padx=5)
        self.resistance_label = ttk.Label(control_frame, text="50.0 Ω", width=10)
        self.resistance_label.grid(row=1, column=2, padx=5)

        # Simulation time
        ttk.Label(control_frame, text="Simulation Time (s):").grid(row=2, column=0, sticky="w", pady=5)
        self.sim_time_var = tk.StringVar(value="0.5")
        ttk.Entry(control_frame, textvariable=self.sim_time_var, width=10).grid(row=2, column=1, sticky="w", padx=5)

        # Control buttons
        button_frame = ttk.Frame(control_frame)
        button_frame.grid(row=3, column=0, columnspan=3, pady=10)

        self.start_button = ttk.Button(button_frame, text="▶ Start", command=self.start_simulation, width=10)
        self.start_button.grid(row=0, column=0, padx=5)

        self.stop_button = ttk.Button(button_frame, text="⬛ Stop", command=self.stop_simulation, width=10, state="disabled")
        self.stop_button.grid(row=0, column=1, padx=5)

        ttk.Button(button_frame, text="⟲ Reset", command=self.reset_simulation, width=10).grid(row=0, column=2, padx=5)

        # Status label
        self.status_label = ttk.Label(control_frame, text=f"Solver: {self.solver_method} | Status: Ready",
                                       foreground="green")
        self.status_label.grid(row=4, column=0, columnspan=3, pady=5)

        # Visualization frame
        viz_frame = ttk.LabelFrame(self.tab_dynamic, text="Real-Time Waveforms", padding=5)
        viz_frame.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)
        viz_frame.grid_rowconfigure(0, weight=1)
        viz_frame.grid_columnconfigure(0, weight=1)

        # Create matplotlib figure with subplots
        self.fig_dynamic = Figure(figsize=(10, 6), dpi=100)
        self.ax_current = self.fig_dynamic.add_subplot(311)
        self.ax_flux = self.fig_dynamic.add_subplot(312)
        self.ax_power = self.fig_dynamic.add_subplot(313)

        self.canvas_dynamic = FigureCanvasTkAgg(self.fig_dynamic, viz_frame)
        self.canvas_dynamic.get_tk_widget().grid(row=0, column=0, sticky="nsew")

        # Configure column weights
        control_frame.grid_columnconfigure(1, weight=1)
        self.tab_dynamic.grid_columnconfigure(0, weight=1)

    def create_analysis_tab(self):
        """Create advanced analysis interface"""
        # Analysis frame
        analysis_frame = ttk.LabelFrame(self.tab_analysis, text="Efficiency Analysis", padding=10)
        analysis_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        analysis_frame.grid_rowconfigure(1, weight=1)
        analysis_frame.grid_columnconfigure(0, weight=1)

        # Controls
        control_frame = ttk.Frame(analysis_frame)
        control_frame.grid(row=0, column=0, sticky="ew", pady=5)

        ttk.Button(control_frame, text="Generate Loss Analysis", command=self.generate_loss_analysis).grid(
            row=0, column=0, padx=5)
        ttk.Button(control_frame, text="Optimization Analysis", command=self.generate_optimization_analysis).grid(
            row=0, column=1, padx=5)
        ttk.Button(control_frame, text="Load Profile Analysis", command=self.generate_load_profile).grid(
            row=0, column=2, padx=5)

        # Visualization
        viz_frame = ttk.Frame(analysis_frame)
        viz_frame.grid(row=1, column=0, sticky="nsew", pady=5)
        viz_frame.grid_rowconfigure(0, weight=1)
        viz_frame.grid_columnconfigure(0, weight=1)

        self.fig_analysis = Figure(figsize=(12, 8), dpi=100)
        self.canvas_analysis = FigureCanvasTkAgg(self.fig_analysis, viz_frame)
        self.canvas_analysis.get_tk_widget().grid(row=0, column=0, sticky="nsew")

    def solve_example_problem(self):
        """Solve the initial example problem"""
        try:
            # Given: efficiency = 90% at both half-load and full-load of 1 kW
            full_load = 1000
            eff_full = 0.90
            eff_half = 0.90

            # Calculate losses
            self.model.calculate_losses_from_efficiency(eff_full, eff_half, full_load)

            # Calculate efficiency at 70% load
            eff_70 = self.model.calculate_efficiency(0.7)

            # Update UI
            self.update_results_display()
            self.plot_efficiency_curve()

            # Show result in message
            result_msg = f"""Example Problem Solution:

Given:
- Full Load: {full_load} W
- Efficiency at Full Load: {eff_full*100}%
- Efficiency at Half Load: {eff_half*100}%

Calculated:
- Core Loss: {self.model.core_loss:.2f} W
- Copper Loss (Full): {self.model.copper_loss_full:.2f} W
- Efficiency at 70% Load: {eff_70*100:.2f}%
            """

            # Display in console
            print(result_msg)

        except Exception as e:
            messagebox.showerror("Error", f"Error solving example: {str(e)}")

    def calculate_losses(self):
        """Calculate core and copper losses from input parameters"""
        try:
            full_load = float(self.full_load_var.get())
            eff_full = float(self.eff_full_var.get()) / 100
            eff_half = float(self.eff_half_var.get()) / 100

            if eff_full <= 0 or eff_full >= 1 or eff_half <= 0 or eff_half >= 1:
                messagebox.showerror("Error", "Efficiency must be between 0 and 100%")
                return

            self.model.calculate_losses_from_efficiency(eff_full, eff_half, full_load)
            self.update_results_display()
            self.plot_efficiency_curve()

            messagebox.showinfo("Success", "Losses calculated successfully!")

        except ValueError:
            messagebox.showerror("Error", "Please enter valid numerical values")

    def update_results_display(self):
        """Update the results labels"""
        self.core_loss_label.config(text=f"Core Loss: {self.model.core_loss:.2f} W")
        self.copper_loss_label.config(text=f"Copper Loss (Full): {self.model.copper_loss_full:.2f} W")

        max_eff_load = self.model.calculate_max_efficiency_load()
        self.max_eff_load_label.config(text=f"Max Efficiency Load: {max_eff_load*100:.1f}%")

    def update_efficiency(self, event=None):
        """Update efficiency display when slider moves"""
        load_fraction = self.load_var.get()
        self.load_value_label.config(text=f"{load_fraction*100:.1f}%")

        efficiency = self.model.calculate_efficiency(load_fraction)
        self.efficiency_result_label.config(text=f"Efficiency: {efficiency*100:.2f}%")

        # Update plot
        self.plot_efficiency_curve()

    def plot_efficiency_curve(self):
        """Plot efficiency vs load curve"""
        self.ax_efficiency.clear()

        # Generate data
        load_fractions = np.linspace(0.1, 1.5, 100)
        efficiencies = [self.model.calculate_efficiency(lf) * 100 for lf in load_fractions]

        # Plot curve
        self.ax_efficiency.plot(load_fractions * 100, efficiencies, 'b-', linewidth=2, label='Efficiency Curve')

        # Mark current point
        current_load = self.load_var.get()
        current_eff = self.model.calculate_efficiency(current_load) * 100
        self.ax_efficiency.plot(current_load * 100, current_eff, 'ro', markersize=10, label='Current Load')

        # Mark maximum efficiency point
        max_eff_load = self.model.calculate_max_efficiency_load()
        if 0.1 <= max_eff_load <= 1.5:
            max_eff = self.model.calculate_efficiency(max_eff_load) * 100
            self.ax_efficiency.plot(max_eff_load * 100, max_eff, 'g^', markersize=12, label='Max Efficiency')

        self.ax_efficiency.set_xlabel('Load (%)', fontsize=10, fontweight='bold')
        self.ax_efficiency.set_ylabel('Efficiency (%)', fontsize=10, fontweight='bold')
        self.ax_efficiency.set_title('Transformer Efficiency vs Load', fontsize=12, fontweight='bold')
        self.ax_efficiency.grid(True, alpha=0.3)
        self.ax_efficiency.legend()
        self.ax_efficiency.set_xlim([0, 160])
        self.ax_efficiency.set_ylim([0, 100])

        self.fig_efficiency.tight_layout()
        self.canvas_efficiency.draw()

    def update_voltage_label(self, event=None):
        """Update voltage label"""
        self.voltage_label.config(text=f"{self.voltage_var.get():.1f} V")

    def update_resistance_label(self, event=None):
        """Update resistance label"""
        self.resistance_label.config(text=f"{self.resistance_var.get():.1f} Ω")

    def start_simulation(self):
        """Start dynamic simulation"""
        if self.simulation_running:
            return

        self.simulation_running = True
        self.start_button.config(state="disabled")
        self.stop_button.config(state="normal")
        self.status_label.config(text=f"Solver: {self.solver_method} | Status: Running...", foreground="orange")

        # Start simulation in separate thread
        self.simulation_thread = threading.Thread(target=self.run_simulation)
        self.simulation_thread.daemon = True
        self.simulation_thread.start()

    def stop_simulation(self):
        """Stop simulation"""
        self.simulation_running = False
        self.start_button.config(state="normal")
        self.stop_button.config(state="disabled")
        self.status_label.config(text=f"Solver: {self.solver_method} | Status: Stopped", foreground="red")

    def reset_simulation(self):
        """Reset simulation"""
        self.stop_simulation()
        self.time_data = []
        self.efficiency_data = []
        self.load_data = []
        self.loss_data = []
        self.current_time = 0

        # Clear plots
        self.ax_current.clear()
        self.ax_flux.clear()
        self.ax_power.clear()
        self.canvas_dynamic.draw()

        self.status_label.config(text=f"Solver: {self.solver_method} | Status: Reset", foreground="green")

    def run_simulation(self):
        """Run the dynamic simulation"""
        try:
            V_in = self.voltage_var.get()
            R_load = self.resistance_var.get()
            sim_time = float(self.sim_time_var.get())

            # Initial conditions [i_p, i_s, phi]
            y0 = [0.0, 0.0, 0.0]
            t_span = [0, sim_time]

            # Choose solver
            if self.solver_method == "Euler":
                t, y = self.model.euler_method(self.model.transformer_dynamics, y0, t_span, args=(V_in, R_load))
            else:
                # Use scipy's solve_ivp with RK45 or RK23
                sol = solve_ivp(self.model.transformer_dynamics, t_span, y0,
                               args=(V_in, R_load), method=self.solver_method,
                               dense_output=True, max_step=0.001)
                t = np.linspace(0, sim_time, 1000)
                y = sol.sol(t).T

            # Extract results
            i_primary = y[:, 0]
            i_secondary = y[:, 1]
            flux = y[:, 2]

            # Calculate power
            power_in = V_in * i_primary
            power_out = (i_secondary ** 2) * R_load

            # Update plots
            self.root.after(0, self.update_dynamic_plots, t, i_primary, i_secondary, flux, power_in, power_out)

            # Update status
            self.root.after(0, lambda: self.status_label.config(
                text=f"Solver: {self.solver_method} | Status: Completed", foreground="green"))

        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Simulation Error", str(e)))
        finally:
            self.simulation_running = False
            self.root.after(0, lambda: self.start_button.config(state="normal"))
            self.root.after(0, lambda: self.stop_button.config(state="disabled"))

    def update_dynamic_plots(self, t, i_p, i_s, flux, power_in, power_out):
        """Update dynamic simulation plots"""
        # Clear previous plots
        self.ax_current.clear()
        self.ax_flux.clear()
        self.ax_power.clear()

        # Plot currents
        self.ax_current.plot(t * 1000, i_p, 'b-', label='Primary Current', linewidth=1.5)
        self.ax_current.plot(t * 1000, i_s, 'r-', label='Secondary Current', linewidth=1.5)
        self.ax_current.set_ylabel('Current (A)', fontweight='bold')
        self.ax_current.set_title('Transformer Currents', fontweight='bold')
        self.ax_current.legend(loc='upper right')
        self.ax_current.grid(True, alpha=0.3)

        # Plot flux
        self.ax_flux.plot(t * 1000, flux, 'g-', linewidth=1.5)
        self.ax_flux.set_ylabel('Flux (Wb)', fontweight='bold')
        self.ax_flux.set_title('Magnetic Flux', fontweight='bold')
        self.ax_flux.grid(True, alpha=0.3)

        # Plot power
        self.ax_power.plot(t * 1000, power_in, 'b-', label='Input Power', linewidth=1.5)
        self.ax_power.plot(t * 1000, power_out, 'r-', label='Output Power', linewidth=1.5)
        self.ax_power.set_xlabel('Time (ms)', fontweight='bold')
        self.ax_power.set_ylabel('Power (W)', fontweight='bold')
        self.ax_power.set_title('Power Flow', fontweight='bold')
        self.ax_power.legend(loc='upper right')
        self.ax_power.grid(True, alpha=0.3)

        self.fig_dynamic.tight_layout()
        self.canvas_dynamic.draw()

    def generate_loss_analysis(self):
        """Generate comprehensive loss analysis"""
        self.fig_analysis.clear()

        # Create subplots
        ax1 = self.fig_analysis.add_subplot(221)
        ax2 = self.fig_analysis.add_subplot(222)
        ax3 = self.fig_analysis.add_subplot(223)
        ax4 = self.fig_analysis.add_subplot(224)

        load_fractions = np.linspace(0.1, 1.5, 100)

        # Plot 1: Losses vs Load
        core_losses = np.full_like(load_fractions, self.model.core_loss)
        copper_losses = self.model.copper_loss_full * (load_fractions ** 2)
        total_losses = core_losses + copper_losses

        ax1.plot(load_fractions * 100, core_losses, 'b-', label='Core Loss', linewidth=2)
        ax1.plot(load_fractions * 100, copper_losses, 'r-', label='Copper Loss', linewidth=2)
        ax1.plot(load_fractions * 100, total_losses, 'k--', label='Total Loss', linewidth=2)
        ax1.set_xlabel('Load (%)', fontweight='bold')
        ax1.set_ylabel('Loss (W)', fontweight='bold')
        ax1.set_title('Losses vs Load', fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot 2: Efficiency vs Load
        efficiencies = [self.model.calculate_efficiency(lf) * 100 for lf in load_fractions]
        ax2.plot(load_fractions * 100, efficiencies, 'g-', linewidth=2)
        ax2.set_xlabel('Load (%)', fontweight='bold')
        ax2.set_ylabel('Efficiency (%)', fontweight='bold')
        ax2.set_title('Efficiency vs Load', fontweight='bold')
        ax2.grid(True, alpha=0.3)

        # Plot 3: Loss Distribution Pie Chart
        max_eff_load = self.model.calculate_max_efficiency_load()
        core_loss_val = self.model.core_loss
        copper_loss_val = self.model.copper_loss_full * (max_eff_load ** 2)

        ax3.pie([core_loss_val, copper_loss_val], labels=['Core Loss', 'Copper Loss'],
                autopct='%1.1f%%', startangle=90, colors=['#ff9999', '#66b3ff'])
        ax3.set_title(f'Loss Distribution at Max Efficiency\n({max_eff_load*100:.1f}% Load)', fontweight='bold')

        # Plot 4: Input vs Output Power
        output_powers = self.model.full_load * load_fractions
        input_powers = output_powers + total_losses

        ax4.plot(load_fractions * 100, input_powers, 'b-', label='Input Power', linewidth=2)
        ax4.plot(load_fractions * 100, output_powers, 'r-', label='Output Power', linewidth=2)
        ax4.fill_between(load_fractions * 100, output_powers, input_powers, alpha=0.3, label='Losses')
        ax4.set_xlabel('Load (%)', fontweight='bold')
        ax4.set_ylabel('Power (W)', fontweight='bold')
        ax4.set_title('Power Flow Analysis', fontweight='bold')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        self.fig_analysis.tight_layout()
        self.canvas_analysis.draw()

    def generate_optimization_analysis(self):
        """Generate optimization analysis"""
        self.fig_analysis.clear()

        # Create subplots
        ax1 = self.fig_analysis.add_subplot(221)
        ax2 = self.fig_analysis.add_subplot(222)
        ax3 = self.fig_analysis.add_subplot(223)
        ax4 = self.fig_analysis.add_subplot(224)

        load_fractions = np.linspace(0.1, 1.5, 100)

        # Plot 1: All Day Efficiency for different load profiles
        # Assume different daily load profiles
        profiles = {
            'Industrial (80% avg)': 0.8,
            'Commercial (60% avg)': 0.6,
            'Residential (40% avg)': 0.4
        }

        for name, avg_load in profiles.items():
            # Simple all-day efficiency calculation
            eff = self.model.calculate_efficiency(avg_load) * 100
            ax1.bar(name, eff, alpha=0.7)

        ax1.set_ylabel('All-Day Efficiency (%)', fontweight='bold')
        ax1.set_title('Efficiency for Different Load Profiles', fontweight='bold')
        ax1.tick_params(axis='x', rotation=15)
        ax1.grid(True, alpha=0.3, axis='y')

        # Plot 2: Efficiency vs Load with optimal operating region
        efficiencies = [self.model.calculate_efficiency(lf) * 100 for lf in load_fractions]
        ax2.plot(load_fractions * 100, efficiencies, 'b-', linewidth=2)

        # Highlight optimal region (within 1% of max efficiency)
        max_eff = max(efficiencies)
        optimal_region = np.array(efficiencies) >= (max_eff - 1)
        ax2.fill_between(load_fractions[optimal_region] * 100,
                          0, np.array(efficiencies)[optimal_region],
                          alpha=0.3, color='green', label='Optimal Region (±1%)')

        ax2.set_xlabel('Load (%)', fontweight='bold')
        ax2.set_ylabel('Efficiency (%)', fontweight='bold')
        ax2.set_title('Optimal Operating Region', fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Plot 3: Cost of losses over time
        # Assume electricity cost of $0.10 per kWh
        electricity_cost = 0.10
        hours_per_year = 8760

        annual_loss_cost = []
        for lf in load_fractions:
            copper_loss = self.model.copper_loss_full * (lf ** 2)
            total_loss = self.model.core_loss + copper_loss
            annual_cost = (total_loss / 1000) * hours_per_year * electricity_cost
            annual_loss_cost.append(annual_cost)

        ax3.plot(load_fractions * 100, annual_loss_cost, 'r-', linewidth=2)
        ax3.set_xlabel('Load (%)', fontweight='bold')
        ax3.set_ylabel('Annual Loss Cost ($)', fontweight='bold')
        ax3.set_title('Annual Cost of Losses\n($0.10/kWh)', fontweight='bold')
        ax3.grid(True, alpha=0.3)

        # Plot 4: Power factor and regulation
        # Simplified voltage regulation calculation
        load_currents = load_fractions * (self.model.full_load / 230)  # Assume 230V
        voltage_drop = load_currents * 2  # Simplified: R_total * I
        voltage_regulation = (voltage_drop / 230) * 100

        ax4.plot(load_fractions * 100, voltage_regulation, 'purple', linewidth=2)
        ax4.set_xlabel('Load (%)', fontweight='bold')
        ax4.set_ylabel('Voltage Regulation (%)', fontweight='bold')
        ax4.set_title('Voltage Regulation vs Load', fontweight='bold')
        ax4.grid(True, alpha=0.3)

        self.fig_analysis.tight_layout()
        self.canvas_analysis.draw()

    def generate_load_profile(self):
        """Generate 24-hour load profile analysis"""
        self.fig_analysis.clear()

        # Create subplots
        ax1 = self.fig_analysis.add_subplot(211)
        ax2 = self.fig_analysis.add_subplot(212)

        # Simulate a typical daily load profile
        hours = np.arange(0, 24, 0.5)

        # Typical commercial/industrial profile
        load_profile = (0.3 + 0.2 * np.sin((hours - 6) * np.pi / 12) +
                       0.15 * np.random.randn(len(hours)) * 0.1)
        load_profile = np.clip(load_profile, 0.2, 0.95)

        # Calculate efficiency for each point
        efficiency_profile = [self.model.calculate_efficiency(lf) * 100 for lf in load_profile]

        # Calculate losses
        core_loss_profile = np.full_like(load_profile, self.model.core_loss)
        copper_loss_profile = self.model.copper_loss_full * (load_profile ** 2)
        total_loss_profile = core_loss_profile + copper_loss_profile

        # Plot 1: Load Profile
        ax1.fill_between(hours, 0, load_profile * 100, alpha=0.5, color='blue', label='Load')
        ax1_twin = ax1.twinx()
        ax1_twin.plot(hours, efficiency_profile, 'r-', linewidth=2, label='Efficiency')

        ax1.set_xlabel('Hour of Day', fontweight='bold')
        ax1.set_ylabel('Load (%)', fontweight='bold', color='blue')
        ax1_twin.set_ylabel('Efficiency (%)', fontweight='bold', color='red')
        ax1.set_title('24-Hour Load and Efficiency Profile', fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim([0, 24])
        ax1.legend(loc='upper left')
        ax1_twin.legend(loc='upper right')

        # Plot 2: Energy and Losses
        energy_output = self.model.full_load * load_profile * 0.5 / 1000  # kWh per interval
        energy_loss = total_loss_profile * 0.5 / 1000

        cumulative_output = np.cumsum(energy_output)
        cumulative_loss = np.cumsum(energy_loss)

        ax2.plot(hours, cumulative_output, 'g-', linewidth=2, label='Cumulative Output Energy')
        ax2.plot(hours, cumulative_loss, 'r-', linewidth=2, label='Cumulative Energy Loss')
        ax2.fill_between(hours, 0, cumulative_loss, alpha=0.3, color='red')

        ax2.set_xlabel('Hour of Day', fontweight='bold')
        ax2.set_ylabel('Energy (kWh)', fontweight='bold')
        ax2.set_title('Cumulative Energy Output and Losses', fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim([0, 24])

        # Add summary text
        total_output = cumulative_output[-1]
        total_loss = cumulative_loss[-1]
        avg_efficiency = (total_output / (total_output + total_loss)) * 100

        summary_text = f'Daily Summary:\nOutput: {total_output:.2f} kWh\nLosses: {total_loss:.2f} kWh\nAvg Efficiency: {avg_efficiency:.2f}%'
        ax2.text(0.02, 0.98, summary_text, transform=ax2.transAxes,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                fontsize=9, fontweight='bold')

        self.fig_analysis.tight_layout()
        self.canvas_analysis.draw()

    def set_solver(self, solver_name):
        """Set the ODE solver method"""
        self.solver_method = solver_name
        self.status_label.config(text=f"Solver: {self.solver_method} | Status: Ready")
        messagebox.showinfo("Solver Changed", f"ODE solver set to: {solver_name}")

    def reset_all(self):
        """Reset all parameters and plots"""
        if messagebox.askyesno("Reset All", "Are you sure you want to reset all parameters?"):
            self.stop_simulation()
            self.model.reset_parameters()
            self.full_load_var.set("1000")
            self.eff_full_var.set("90")
            self.eff_half_var.set("90")
            self.load_var.set(0.7)
            self.voltage_var.set(230)
            self.resistance_var.set(50)
            self.sim_time_var.set("0.5")

            # Clear all plots
            for ax in [self.ax_efficiency, self.ax_current, self.ax_flux, self.ax_power]:
                ax.clear()

            self.fig_analysis.clear()

            self.canvas_efficiency.draw()
            self.canvas_dynamic.draw()
            self.canvas_analysis.draw()

            # Recalculate with default values
            self.solve_example_problem()

    def show_about(self):
        """Show about dialog"""
        about_text = """Advanced Transformer Efficiency Calculator & Simulator

Version: 2.0

Features:
• Transformer efficiency calculation
• Dynamic differential equation simulation
• Multiple ODE solvers (RK45, RK23, Euler)
• Real-time waveform visualization
• Comprehensive loss analysis
• Load profile optimization
• Professional electrical engineering tools

Developed for practical electrical engineering applications
        """
        messagebox.showinfo("About", about_text)

    def on_resize(self, event):
        """Handle window resize event"""
        # Redraw canvases to fit new window size
        try:
            self.canvas_efficiency.draw()
            self.canvas_dynamic.draw()
            self.canvas_analysis.draw()
        except:
            pass


def main():
    """Main function to run the application"""
    root = tk.Tk()
    app = TransformerSimulator(root)

    # Print solution to console
    print("=" * 70)
    print("TRANSFORMER EFFICIENCY PROBLEM SOLUTION")
    print("=" * 70)
    print("\nGiven:")
    print("- Single-phase transformer at unity power factor")
    print("- Efficiency = 90% at both half-load and full-load")
    print("- Full load = 1 kW")
    print("\nFind: Efficiency at 70% of full-load")
    print("\nSolution:")
    print(f"Core Loss: {app.model.core_loss:.2f} W")
    print(f"Copper Loss (Full Load): {app.model.copper_loss_full:.2f} W")

    eff_70 = app.model.calculate_efficiency(0.7) * 100
    print(f"\nEfficiency at 70% load: {eff_70:.2f}%")
    print("=" * 70)

    root.mainloop()


if __name__ == "__main__":
    main()
