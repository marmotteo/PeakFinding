import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from scipy.signal import find_peaks, peak_widths, savgol_filter
from scipy.optimize import curve_fit
import os
import threading

class PeakFindingGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Chromatographic Peak Finding Tool")
        self.root.geometry("1400x800")
        
        # Data storage
        self.data = None
        self.file_path = None
        self.peak_results = None
        
        # Create main frame
        main_frame = ttk.Frame(root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        root.columnconfigure(0, weight=1)
        root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)  # Left panel (file loading)
        main_frame.columnconfigure(1, weight=1)  # Middle panel (parameters)
        main_frame.columnconfigure(2, weight=1)  # Right panel (small results)
        main_frame.rowconfigure(0, weight=0)     # Top row for controls
        main_frame.rowconfigure(1, weight=0)     # Plot toolbar row
        main_frame.rowconfigure(2, weight=1)     # Bottom row for plotting
        
        self.create_top_panels(main_frame)
        self.create_plot_toolbar(main_frame)
        self.create_plotting_panel(main_frame)
        
    def create_top_panels(self, parent):
        """Create file loading, parameters, and small results panels at the top"""
        # File loading panel (top-left) - KEEP INTACT
        left_frame = ttk.LabelFrame(parent, text="File Loading", padding="10")
        left_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 5), pady=(0, 10))
        
        # File selection
        ttk.Label(left_frame, text="Select Data File:").grid(row=0, column=0, sticky=tk.W, pady=(0, 5))
        
        file_frame = ttk.Frame(left_frame)
        file_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        file_frame.columnconfigure(0, weight=1)
        
        self.file_path_var = tk.StringVar()
        file_entry = ttk.Entry(file_frame, textvariable=self.file_path_var, state="readonly")
        file_entry.grid(row=0, column=0, sticky=(tk.W, tk.E), padx=(0, 5))
        
        browse_btn = ttk.Button(file_frame, text="Browse", command=self.browse_file)
        browse_btn.grid(row=0, column=1)
        
        # File list
        ttk.Label(left_frame, text="Recent Files:").grid(row=2, column=0, sticky=tk.W, pady=(10, 5))
        
        # Create listbox with scrollbar
        list_frame = ttk.Frame(left_frame)
        list_frame.grid(row=3, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        list_frame.columnconfigure(0, weight=1)
        list_frame.rowconfigure(0, weight=1)
        
        self.file_listbox = tk.Listbox(list_frame, height=6)
        self.file_listbox.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.file_listbox.bind('<<ListboxSelect>>', self.on_file_select)
        
        list_scrollbar = ttk.Scrollbar(list_frame, orient="vertical", command=self.file_listbox.yview)
        list_scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        self.file_listbox.configure(yscrollcommand=list_scrollbar.set)
        
        # Load recent files
        self.load_recent_files()
        
        # File info
        ttk.Label(left_frame, text="File Information:").grid(row=4, column=0, sticky=tk.W, pady=(10, 5))
        
        self.file_info_text = scrolledtext.ScrolledText(left_frame, height=4, width=40)
        self.file_info_text.grid(row=5, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        
        # Configure grid weights for left panel
        left_frame.columnconfigure(0, weight=1)
        left_frame.rowconfigure(3, weight=1)
        left_frame.rowconfigure(5, weight=1)
        
        # Parameters panel (top-middle)
        params_frame = ttk.LabelFrame(parent, text="Peak Finding Parameters", padding="10")
        params_frame.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5, pady=(0, 10))
        params_frame.columnconfigure(1, weight=1)
        
        # Parameter variables
        self.baseline_poly_order = tk.IntVar(value=5)
        self.savgol_window = tk.IntVar(value=15)
        self.savgol_poly_order = tk.IntVar(value=3)
        self.peak_height_threshold_factor = tk.DoubleVar(value=3.0)
        self.peak_distance_points = tk.IntVar(value=20)
        self.snr_threshold = tk.DoubleVar(value=10.0)
        self.peak_width_rel_height = tk.DoubleVar(value=0.5)
        
        # Create parameter controls
        params = [
            ("Baseline Polynomial Order:", self.baseline_poly_order, 1, 10),
            ("Savgol Window Size:", self.savgol_window, 5, 51),
            ("Savgol Polynomial Order:", self.savgol_poly_order, 1, 5),
            ("Peak Height Threshold Factor:", self.peak_height_threshold_factor, 1.0, 10.0),
            ("Peak Distance (points):", self.peak_distance_points, 5, 100),
            ("SNR Threshold:", self.snr_threshold, 1.0, 50.0),
            ("Peak Width Relative Height:", self.peak_width_rel_height, 0.1, 1.0)
        ]
        
        for i, (label, var, min_val, max_val) in enumerate(params):
            ttk.Label(params_frame, text=label).grid(row=i, column=0, sticky=tk.W, pady=2)
            
            if isinstance(var, tk.IntVar):
                spinbox = ttk.Spinbox(params_frame, from_=min_val, to=max_val, textvariable=var, width=10)
            else:
                spinbox = ttk.Spinbox(params_frame, from_=min_val, to=max_val, increment=0.1, 
                                    textvariable=var, width=10, format="%.1f")
            spinbox.grid(row=i, column=1, sticky=tk.W, padx=(10, 0), pady=2)
        
        # Control buttons
        button_frame = ttk.Frame(params_frame)
        button_frame.grid(row=len(params), column=0, columnspan=2, pady=(10, 0))
        
        ttk.Button(button_frame, text="Reset to Defaults", command=self.reset_parameters).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(button_frame, text="Run Analysis", command=self.run_analysis).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Save Results", command=self.save_results).pack(side=tk.LEFT, padx=5)
        
        # Small Results panel (top-right)
        results_frame = ttk.LabelFrame(parent, text="Results", padding="10")
        results_frame.grid(row=0, column=2, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(5, 0), pady=(0, 10))
        results_frame.columnconfigure(0, weight=1)
        results_frame.rowconfigure(0, weight=1)
        
        # Create treeview for results (smaller)
        columns = ('Peak', 'Time', 'Height', 'SNR')
        self.results_tree = ttk.Treeview(results_frame, columns=columns, show='headings', height=8)
        
        for col in columns:
            self.results_tree.heading(col, text=col)
            self.results_tree.column(col, width=60)
        
        self.results_tree.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Scrollbar for treeview
        tree_scrollbar = ttk.Scrollbar(results_frame, orient="vertical", command=self.results_tree.yview)
        tree_scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        self.results_tree.configure(yscrollcommand=tree_scrollbar.set)
        
    def create_plot_toolbar(self, parent):
        """Create plot function toolbar"""
        toolbar_frame = ttk.LabelFrame(parent, text="Plot Controls", padding="5")
        toolbar_frame.grid(row=1, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 5))
        
        # Plot control variables
        self.grid_visible = tk.BooleanVar(value=True)
        self.legend_visible = tk.BooleanVar(value=True)
        
        # Create toolbar buttons
        ttk.Button(toolbar_frame, text="🔍 Zoom", command=self.toggle_zoom).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar_frame, text="✋ Pan", command=self.toggle_pan).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar_frame, text="🏠 Home", command=self.reset_view).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar_frame, text="⬅️ Back", command=self.go_back).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar_frame, text="➡️ Forward", command=self.go_forward).pack(side=tk.LEFT, padx=2)
        
        ttk.Separator(toolbar_frame, orient='vertical').pack(side=tk.LEFT, fill='y', padx=5)
        
        ttk.Checkbutton(toolbar_frame, text="Grid", variable=self.grid_visible, command=self.toggle_grid).pack(side=tk.LEFT, padx=2)
        ttk.Checkbutton(toolbar_frame, text="Legend", variable=self.legend_visible, command=self.toggle_legend).pack(side=tk.LEFT, padx=2)
        
        ttk.Separator(toolbar_frame, orient='vertical').pack(side=tk.LEFT, fill='y', padx=5)
        
        ttk.Button(toolbar_frame, text="💾 Save Plot", command=self.save_plot).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar_frame, text="🔄 Refresh", command=self.refresh_plot).pack(side=tk.LEFT, padx=2)
        
        # Add plot type selection
        ttk.Label(toolbar_frame, text="Plot Type:").pack(side=tk.LEFT, padx=(10, 2))
        self.plot_type = tk.StringVar(value="line")
        plot_type_combo = ttk.Combobox(toolbar_frame, textvariable=self.plot_type, 
                                      values=["line", "scatter", "both"], width=8, state="readonly")
        plot_type_combo.pack(side=tk.LEFT, padx=2)
        plot_type_combo.bind('<<ComboboxSelected>>', self.on_plot_type_change)
        
    def create_plotting_panel(self, parent):
        """Create main plotting area at the bottom spanning full width"""
        # Plotting panel
        plot_frame = ttk.LabelFrame(parent, text="Chromatogram with Peak Detection", padding="10")
        plot_frame.grid(row=2, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S))
        plot_frame.columnconfigure(0, weight=1)
        plot_frame.rowconfigure(1, weight=1)
        
        # Create matplotlib figure
        self.fig, self.ax = plt.subplots(1, 1, figsize=(14, 6))
        
        # Create a frame for the canvas to avoid geometry manager conflicts
        canvas_frame = ttk.Frame(plot_frame)
        canvas_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        self.canvas = FigureCanvasTkAgg(self.fig, canvas_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Create matplotlib navigation toolbar in a separate frame
        toolbar_frame = ttk.Frame(plot_frame)
        toolbar_frame.grid(row=0, column=0, sticky=(tk.W, tk.E))
        
        self.nav_toolbar = NavigationToolbar2Tk(self.canvas, toolbar_frame)
        
        # Initialize empty plot
        self.ax.set_title('Chromatogram - Load a file to display data')
        self.ax.set_xlabel('Time (minutes)')
        self.ax.set_ylabel('Intensity')
        self.ax.grid(True, alpha=0.3)
        self.canvas.draw()
        
    # Plot toolbar functions
    def toggle_zoom(self):
        """Toggle zoom mode"""
        self.nav_toolbar.zoom()
        
    def toggle_pan(self):
        """Toggle pan mode"""
        self.nav_toolbar.pan()
        
    def reset_view(self):
        """Reset view to home"""
        self.nav_toolbar.home()
        
    def go_back(self):
        """Go back in view history"""
        self.nav_toolbar.back()
        
    def go_forward(self):
        """Go forward in view history"""
        self.nav_toolbar.forward()
        
    def toggle_grid(self):
        """Toggle grid visibility"""
        self.ax.grid(self.grid_visible.get(), alpha=0.3)
        self.canvas.draw()
        
    def toggle_legend(self):
        """Toggle legend visibility"""
        legend = self.ax.get_legend()
        if legend:
            legend.set_visible(self.legend_visible.get())
            self.canvas.draw()
            
    def save_plot(self):
        """Save current plot to file"""
        filename = filedialog.asksaveasfilename(
            title="Save Plot",
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("PDF files", "*.pdf"), 
                      ("SVG files", "*.svg"), ("All files", "*.*")]
        )
        
        if filename:
            try:
                self.fig.savefig(filename, dpi=300, bbox_inches='tight')
                messagebox.showinfo("Success", f"Plot saved to {filename}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save plot: {str(e)}")
                
    def refresh_plot(self):
        """Refresh the current plot"""
        if self.peak_results:
            self.update_plot_with_peaks(self.peak_results)
        elif self.data is not None:
            self.plot_file_content()
            
    def on_plot_type_change(self, event=None):
        """Handle plot type change"""
        self.refresh_plot()
        
    def plot_file_content(self):
        """Plot the current file content"""
        if self.data is None:
            return
            
        self.ax.clear()
        
        # Plot based on selected type
        plot_type = self.plot_type.get()
        
        if plot_type == "line":
            self.ax.plot(self.data['Time'], self.data['Intensity'], 'b-', linewidth=1, label='Raw Signal')
        elif plot_type == "scatter":
            self.ax.scatter(self.data['Time'], self.data['Intensity'], c='blue', s=1, alpha=0.6, label='Raw Signal')
        else:  # both
            self.ax.plot(self.data['Time'], self.data['Intensity'], 'b-', linewidth=0.5, alpha=0.7, label='Raw Signal')
            self.ax.scatter(self.data['Time'], self.data['Intensity'], c='blue', s=0.5, alpha=0.3)
        
        self.ax.set_title(f'Chromatogram: {os.path.basename(self.file_path) if self.file_path else "No file loaded"}')
        self.ax.set_xlabel('Time (minutes)')
        self.ax.set_ylabel('Intensity')
        
        # Set axis tight
        self.ax.axis('tight')
        
        if self.legend_visible.get():
            self.ax.legend()
        self.ax.grid(self.grid_visible.get(), alpha=0.3)
        
        self.canvas.draw()
        
    def update_plot_with_peaks(self, results):
        """Update the plot to show file content with detected peaks"""
        self.ax.clear()
        
        data = results['data']
        peaks = results['peaks']
        peak_indices = results['peak_indices']
        
        # Plot based on selected type
        plot_type = self.plot_type.get()
        
        if plot_type == "line":
            self.ax.plot(data['Time'], data['Intensity_Original'], 'lightgray', linewidth=1, alpha=0.7, label='Original Signal')
            self.ax.plot(data['Time'], data['Baseline'], 'r--', linewidth=1, alpha=0.8, label='Baseline')
            self.ax.plot(data['Time'], data['Intensity_Smoothed'], 'b-', linewidth=1.5, label='Smoothed Signal')
        elif plot_type == "scatter":
            self.ax.scatter(data['Time'], data['Intensity_Original'], c='lightgray', s=1, alpha=0.7, label='Original Signal')
            self.ax.plot(data['Time'], data['Baseline'], 'r--', linewidth=1, alpha=0.8, label='Baseline')
            self.ax.scatter(data['Time'], data['Intensity_Smoothed'], c='blue', s=1, alpha=0.8, label='Smoothed Signal')
        else:  # both
            self.ax.plot(data['Time'], data['Intensity_Original'], 'lightgray', linewidth=0.5, alpha=0.5, label='Original Signal')
            self.ax.scatter(data['Time'], data['Intensity_Original'], c='lightgray', s=0.3, alpha=0.3)
            self.ax.plot(data['Time'], data['Baseline'], 'r--', linewidth=1, alpha=0.8, label='Baseline')
            self.ax.plot(data['Time'], data['Intensity_Smoothed'], 'b-', linewidth=1, alpha=0.8, label='Smoothed Signal')
            self.ax.scatter(data['Time'], data['Intensity_Smoothed'], c='blue', s=0.5, alpha=0.5)
        
        # Mark detected peaks
        if len(peak_indices) > 0:
            peak_times = data['Time'].iloc[peak_indices].values
            peak_heights = data['Intensity_Smoothed'].iloc[peak_indices].values
            
            self.ax.plot(peak_times, peak_heights, 'ro', markersize=8, label=f'Detected Peaks ({len(peak_indices)})')
            
            # Add peak labels
            for i, (time, height) in enumerate(zip(peak_times, peak_heights)):
                snr = peaks.iloc[i]['SNR'] if i < len(peaks) else 0
                self.ax.annotate(f'P{i+1}\nSNR:{snr:.1f}', 
                               xy=(time, height), 
                               xytext=(5, 10), textcoords='offset points',
                               fontsize=8, ha='left', va='bottom',
                               bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.7))
        
        self.ax.set_title(f'Chromatogram with Peak Detection: {os.path.basename(self.file_path) if self.file_path else "No file loaded"}')
        self.ax.set_xlabel('Time (minutes)')
        self.ax.set_ylabel('Intensity')
        
        # Set axis tight
        self.ax.axis('tight')
        
        if self.legend_visible.get():
            self.ax.legend()
        self.ax.grid(self.grid_visible.get(), alpha=0.3)
        
        self.canvas.draw()

    def browse_file(self):
        """Open file dialog to select data file"""
        filetypes = [
            ('Text files', '*.txt'),
            ('CSV files', '*.csv'),
            ('CDF files', '*.cdf'),
            ('All files', '*.*')
        ]
        
        filename = filedialog.askopenfilename(
            title="Select Data File",
            filetypes=filetypes
        )
        
        if filename:
            self.file_path_var.set(filename)
            self.file_path = filename
            self.load_file_info()
            self.add_to_recent_files(filename)
            self.plot_file_content()  # Plot file content immediately after loading
            
    def load_recent_files(self):
        """Load recent files from the current directory"""
        try:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            supported_extensions = ['.txt', '.csv', '.cdf']
            
            files = []
            for file in os.listdir(current_dir):
                if any(file.lower().endswith(ext) for ext in supported_extensions):
                    files.append(os.path.join(current_dir, file))
            
            self.file_listbox.delete(0, tk.END)
            for file in sorted(files):
                self.file_listbox.insert(tk.END, os.path.basename(file))
                
        except Exception as e:
            print(f"Error loading recent files: {e}")
            
    def on_file_select(self, event):
        """Handle file selection from listbox"""
        selection = self.file_listbox.curselection()
        if selection:
            filename = self.file_listbox.get(selection[0])
            current_dir = os.path.dirname(os.path.abspath(__file__))
            full_path = os.path.join(current_dir, filename)
            
            if os.path.exists(full_path):
                self.file_path_var.set(full_path)
                self.file_path = full_path
                self.load_file_info()
                self.plot_file_content()  # Plot file content immediately after selection
                
    def add_to_recent_files(self, filepath):
        """Add file to recent files list"""
        filename = os.path.basename(filepath)
        
        # Remove if already exists
        for i in range(self.file_listbox.size()):
            if self.file_listbox.get(i) == filename:
                self.file_listbox.delete(i)
                break
        
        # Add to top
        self.file_listbox.insert(0, filename)
        
    def load_file_info(self):
        """Load and display file information"""
        if not self.file_path or not os.path.exists(self.file_path):
            return
            
        try:
            # Try to read the file
            if self.file_path.lower().endswith('.csv'):
                data = pd.read_csv(self.file_path)
            else:
                # Assume tab-separated for .txt and .cdf files
                data = pd.read_csv(self.file_path, sep='\t', header=None)
                data.columns = ['Time', 'Intensity']
            
            self.data = data
            
            # Display file info
            info = f"File: {os.path.basename(self.file_path)}\n"
            info += f"Size: {os.path.getsize(self.file_path)} bytes\n"
            info += f"Rows: {len(data)}\n"
            info += f"Columns: {len(data.columns)}\n"
            info += f"Column names: {list(data.columns)}\n\n"
            info += "First 5 rows:\n"
            info += data.head().to_string()
            
            self.file_info_text.delete(1.0, tk.END)
            self.file_info_text.insert(1.0, info)
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load file: {str(e)}")
            
    def reset_parameters(self):
        """Reset parameters to default values"""
        self.baseline_poly_order.set(5)
        self.savgol_window.set(15)
        self.savgol_poly_order.set(3)
        self.peak_height_threshold_factor.set(3.0)
        self.peak_distance_points.set(20)
        self.snr_threshold.set(10.0)
        self.peak_width_rel_height.set(0.5)
        
    def run_analysis(self):
        """Run peak finding analysis"""
        if self.data is None:
            messagebox.showwarning("Warning", "Please load a data file first.")
            return
            
        # Run analysis in a separate thread to prevent GUI freezing
        threading.Thread(target=self._run_analysis_thread, daemon=True).start()
        
    def _run_analysis_thread(self):
        """Run analysis in separate thread"""
        try:
            # Use the same peak finding logic from the original script
            results = self.find_chromatographic_peaks()
            
            # Update GUI in main thread
            self.root.after(0, self._update_results, results)
            
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Error", f"Analysis failed: {str(e)}"))
            
    def find_chromatographic_peaks(self):
        """Peak finding algorithm (adapted from original script)"""
        data = self.data.copy()
        
        # Ensure we have the right column names
        if 'Time' not in data.columns or 'Intensity' not in data.columns:
            if len(data.columns) >= 2:
                data.columns = ['Time', 'Intensity']
            else:
                raise ValueError("Data must have at least 2 columns (Time and Intensity)")
        
        data['Intensity_Original'] = data['Intensity'].copy()
        
        # Baseline correction
        def polynomial_baseline(x, *coeffs):
            return np.polyval(coeffs, x)
        
        x_data = np.arange(len(data['Intensity']))
        initial_guess = [0] * (self.baseline_poly_order.get() + 1)
        
        try:
            popt, _ = curve_fit(polynomial_baseline, x_data, data['Intensity'], p0=initial_guess)
            fitted_baseline = polynomial_baseline(x_data, *popt)
        except:
            fitted_baseline = np.full(len(data['Intensity']), data['Intensity'].min())
        
        intensity_corrected = data['Intensity'] - fitted_baseline
        data['Intensity_Corrected'] = intensity_corrected
        data['Baseline'] = fitted_baseline
        
        # Noise reduction
        window = self.savgol_window.get()
        if window % 2 == 0:
            window += 1  # Ensure odd number
        
        intensity_smoothed = savgol_filter(
            data['Intensity_Corrected'].values, 
            window_length=min(window, len(data) - 1), 
            polyorder=min(self.savgol_poly_order.get(), window - 1)
        )
        data['Intensity_Smoothed'] = intensity_smoothed
        
        # Noise estimation
        num_points = len(data['Intensity_Smoothed'])
        baseline_segment_end_idx = min(num_points, max(int(num_points * 0.1), window * 2))
        
        noise_segment = data['Intensity_Smoothed'].iloc[:baseline_segment_end_idx]
        noise_level = noise_segment.std() if len(noise_segment) > 1 else 1e-9
        
        if noise_level == 0 or np.isnan(noise_level) or np.isinf(noise_level):
            noise_level = 1e-9
        
        # Peak detection
        peaks_indices, properties = find_peaks(
            data['Intensity_Smoothed'], 
            height=self.peak_height_threshold_factor.get() * noise_level, 
            distance=self.peak_distance_points.get()
        )
        
        # SNR calculation and filtering
        if len(peaks_indices) > 0:
            peak_heights_for_snr = data['Intensity_Corrected'].iloc[peaks_indices].values
            snr_values = peak_heights_for_snr / noise_level
            snr_values = np.nan_to_num(snr_values, nan=0.0, posinf=1e6, neginf=-1e6)
            
            snr_filter_mask = snr_values >= self.snr_threshold.get()
            significant_peaks_indices = peaks_indices[snr_filter_mask]
            significant_peak_heights_corrected = peak_heights_for_snr[snr_filter_mask]
            significant_snr_values = snr_values[snr_filter_mask]
        else:
            significant_peaks_indices = np.array([])
            significant_peak_heights_corrected = np.array([])
            significant_snr_values = np.array([])
        
        # Determine peak boundaries
        peak_info_list = []
        if len(significant_peaks_indices) > 0:
            widths, width_heights, left_ips, right_ips = peak_widths(
                data['Intensity_Smoothed'], significant_peaks_indices, 
                rel_height=self.peak_width_rel_height.get()
            )
            
            time_indices = np.arange(len(data['Time']))
            start_times = np.interp(left_ips, time_indices, data['Time'])
            end_times = np.interp(right_ips, time_indices, data['Time'])
            
            for i in range(len(significant_peaks_indices)):
                peak_idx = significant_peaks_indices[i]
                peak_info_list.append({
                    'Peak_Number': i + 1,
                    'Time': data['Time'].iloc[peak_idx],
                    'Height': significant_peak_heights_corrected[i],
                    'Start_Time': start_times[i],
                    'End_Time': end_times[i],
                    'SNR': significant_snr_values[i]
                })
        
        peak_table = pd.DataFrame(peak_info_list)
        
        return {
            'data': data,
            'peaks': peak_table,
            'peak_indices': significant_peaks_indices,
            'noise_level': noise_level
        }
        
    def _update_results(self, results):
        """Update GUI with analysis results"""
        self.peak_results = results
        
        # Update results table (smaller version)
        for item in self.results_tree.get_children():
            self.results_tree.delete(item)
            
        for _, row in results['peaks'].iterrows():
            self.results_tree.insert('', 'end', values=(
                int(row['Peak_Number']),
                f"{row['Time']:.3f}",
                f"{row['Height']:.2f}",
                f"{row['SNR']:.1f}"
            ))
        
        # Update plot with peaks
        self.update_plot_with_peaks(results)
        
        messagebox.showinfo("Success", f"Analysis complete! Found {len(results['peaks'])} significant peaks.")
        
    def save_results(self):
        """Save analysis results to CSV file"""
        if self.peak_results is None:
            messagebox.showwarning("Warning", "No results to save. Please run analysis first.")
            return
            
        filename = filedialog.asksaveasfilename(
            title="Save Results",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        
        if filename:
            try:
                self.peak_results['peaks'].to_csv(filename, index=False, float_format='%.4f')
                messagebox.showinfo("Success", f"Results saved to {filename}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save results: {str(e)}")

def main():
    root = tk.Tk()
    app = PeakFindingGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()