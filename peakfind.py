import pandas as pd
import numpy as np
from scipy.signal import find_peaks, peak_widths
import matplotlib.pyplot as plt
import os

# 获取脚本所在目录并构建数据文件的完整路径
script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, 'AS301-24-10-01.dx_FID1A.txt')

# 检查文件是否存在
if not os.path.exists(file_path):
    print(f"错误：找不到数据文件 {file_path}")
    print(f"请确保文件 'AS301-24-10-01.dx_FID1A.txt' 在脚本同一目录下")
    exit(1)

# Load the data from the file
data = pd.read_csv(file_path, sep='\t', header=None, names=['Time', 'Intensity'])

# Convert columns to numeric, coercing errors
data['Time'] = pd.to_numeric(data['Time'], errors='coerce')
data['Intensity'] = pd.to_numeric(data['Intensity'], errors='coerce')

# Drop rows with NaN values that may have resulted from coercion
data.dropna(inplace=True)

# 1. Noise Estimation
# Estimate noise from the first part of the signal (e.g., the first minute)
baseline_data = data[data['Time'] <= 1.0]
noise_level = baseline_data['Intensity'].std()
print(f"估计的噪声水平: {noise_level:.6f}")

# 2. Peak Detection
# Find peaks in the intensity signal.
# We can set a minimum height for a peak to be considered, based on the noise level.
# A reasonable starting point is a height of at least 3 times the noise level.
peaks, properties = find_peaks(data['Intensity'], height=3*noise_level, distance=100)
print(f"检测到 {len(peaks)} 个潜在峰")

# 3. S/N Calculation and Filtering
# Calculate Signal-to-Noise for each peak
peak_heights = properties['peak_heights']
snr = peak_heights / noise_level

# Filter peaks with S/N > 10
snr_filter = snr > 10
significant_peaks = peaks[snr_filter]
significant_peak_heights = peak_heights[snr_filter]
significant_snr = snr[snr_filter]

print(f"信噪比大于10的峰数量: {len(significant_peaks)}")

# 4. Determine Peak Boundaries
if len(significant_peaks) > 0:
    # Calculate the widths of the significant peaks to find their start and end times
    widths, width_heights, left_ips, right_ips = peak_widths(data['Intensity'], significant_peaks, rel_height=0.95)
    
    # Convert interpolated positions to time values.
    start_times = np.interp(left_ips, np.arange(len(data['Time'])), data['Time'])
    end_times = np.interp(right_ips, np.arange(len(data['Time'])), data['Time'])
    
    # Prepare the results
    peak_info_list = []
    for i in range(len(significant_peaks)):
        peak_index = significant_peaks[i]
        peak_time = data['Time'].iloc[peak_index]
        peak_height = significant_peak_heights[i]
        start_time = start_times[i]
        end_time = end_times[i]
        snr_value = significant_snr[i]
    
        peak_info_list.append({
            'Peak_Number': i + 1,
            'Time': peak_time,
            'Height': peak_height,
            'Start_Time': start_time,
            'End_Time': end_time,
            'SNR': snr_value
        })
    
    # Convert to DataFrame for nice printing
    peak_table = pd.DataFrame(peak_info_list)
    
    # Print the results
    print("\n=== 检测到的峰（信噪比 > 10）===")
    print(peak_table.to_string(index=False, float_format='%.4f'))
    
    # Save results to CSV
    output_csv = os.path.join(script_dir, 'detected_peaks.csv')
    peak_table.to_csv(output_csv, index=False, encoding='utf-8-sig')
    print(f"\n结果已保存到: {output_csv}")
    
    # Generate and save a plot for visualization
    plt.figure(figsize=(15, 8))
    
    # 主图
    plt.subplot(2, 1, 1)
    plt.plot(data['Time'], data['Intensity'], label='信号', linewidth=1)
    plt.plot(data['Time'].iloc[significant_peaks], significant_peak_heights, 'x', 
             color='red', markersize=8, label=f'检测到的峰 (信噪比 > 10, 共{len(significant_peaks)}个)')
    plt.hlines(y=width_heights, xmin=start_times, xmax=end_times, 
               color="green", linewidth=2, label='峰宽')
    
    # 标注峰编号
    for i, peak_idx in enumerate(significant_peaks):
        peak_time = data['Time'].iloc[peak_idx]
        peak_height = significant_peak_heights[i]
        plt.annotate(f'峰{i+1}\nSNR:{significant_snr[i]:.1f}', 
                    xy=(peak_time, peak_height), 
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=8, ha='left')
    
    plt.xlabel('时间 (分钟)')
    plt.ylabel('强度')
    plt.title('色谱图及检测到的峰')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 局部放大图（前10分钟）
    plt.subplot(2, 1, 2)
    mask = data['Time'] <= 10
    plt.plot(data['Time'][mask], data['Intensity'][mask], label='信号', linewidth=1)
    
    # 标记前10分钟内的峰
    early_peaks_mask = data['Time'].iloc[significant_peaks] <= 10
    if np.any(early_peaks_mask):
        early_peaks = significant_peaks[early_peaks_mask]
        early_heights = significant_peak_heights[early_peaks_mask]
        early_snr = significant_snr[early_peaks_mask]
        
        plt.plot(data['Time'].iloc[early_peaks], early_heights, 'x', 
                 color='red', markersize=8, label='检测到的峰')
        
        for i, peak_idx in enumerate(early_peaks):
            peak_time = data['Time'].iloc[peak_idx]
            peak_height = early_heights[i]
            plt.annotate(f'峰{np.where(significant_peaks == peak_idx)[0][0] + 1}', 
                        xy=(peak_time, peak_height), 
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=8, ha='left')
    
    plt.xlabel('时间 (分钟)')
    plt.ylabel('强度')
    plt.title('色谱图前10分钟放大')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存图片
    output_png = os.path.join(script_dir, 'chromatogram_with_peaks.png')
    plt.savefig(output_png, dpi=300, bbox_inches='tight')
    print(f"图片已保存到: {output_png}")
    plt.show()
    
else:
    print("未检测到符合条件的峰（信噪比 > 10）")