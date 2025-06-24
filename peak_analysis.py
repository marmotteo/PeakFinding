import numpy as np
import pandas as pd
from scipy.signal import find_peaks, peak_widths, savgol_filter
import matplotlib.pyplot as plt
import os

def analyze_peaks(file_path, snr_threshold=10, min_peak_height_ratio=0.01):
    """
    分析色谱数据中的峰
    
    参数:
    file_path: 数据文件路径
    snr_threshold: 信噪比阈值
    min_peak_height_ratio: 最小峰高相对于最大峰的比例
    
    返回:
    peaks_info: 包含峰信息的列表
    """
    
    # 读取数据
    data = pd.read_csv(file_path, sep='\t', header=None, names=['time', 'intensity'])
    time = data['time'].values
    intensity = data['intensity'].values
    
    # 数据平滑处理，减少噪声
    intensity_smooth = savgol_filter(intensity, window_length=11, polyorder=3)
    
    # 计算基线噪声（使用前1000个点的标准差作为噪声估计）
    baseline_region = intensity_smooth[:1000]
    baseline_noise = np.std(baseline_region)
    baseline_mean = np.mean(baseline_region)
    
    print(f"估计的基线噪声: {baseline_noise:.6f}")
    print(f"基线平均值: {baseline_mean:.6f}")
    
    # 计算数据的整体统计信息
    max_intensity = np.max(intensity_smooth)
    print(f"最大强度值: {max_intensity:.6f}")
    
    # 动态设置峰检测参数
    # 最小峰高：基线 + 5倍噪声，但不能低于最大值的一定比例
    min_height_noise = baseline_mean + 5 * baseline_noise
    min_height_relative = max_intensity * min_peak_height_ratio
    min_height = max(min_height_noise, min_height_relative)
    
    print(f"最小峰高阈值: {min_height:.6f}")
    
    # 寻找峰，使用更严格的参数
    peaks, properties = find_peaks(intensity_smooth, 
                                 height=min_height,
                                 distance=50,  # 增加最小峰间距离
                                 prominence=max(baseline_noise * 5, max_intensity * 0.005),  # 更严格的突出度要求
                                 width=3)  # 最小峰宽
    
    print(f"找到 {len(peaks)} 个潜在峰")
    
    if len(peaks) == 0:
        print("未检测到任何峰")
        return [], time, intensity, []
    
    # 计算峰宽
    try:
        widths, width_heights, left_ips, right_ips = peak_widths(intensity_smooth, peaks, rel_height=0.5)
    except:
        print("峰宽计算失败，使用简化方法")
        widths = np.ones(len(peaks)) * 10
        left_ips = peaks - 5
        right_ips = peaks + 5
    
    # 分析每个峰
    peaks_info = []
    
    for i, peak_idx in enumerate(peaks):
        peak_time = time[peak_idx]
        peak_height = intensity_smooth[peak_idx]
        
        # 计算峰的区间
        if i < len(left_ips):
            left_idx = max(0, int(left_ips[i]))
            right_idx = min(len(time)-1, int(right_ips[i]))
        else:
            left_idx = max(0, peak_idx - 10)
            right_idx = min(len(time)-1, peak_idx + 10)
        
        start_time = time[left_idx]
        end_time = time[right_idx]
        
        # 更准确的基线计算
        # 在峰的两侧寻找局部最小值作为基线
        left_baseline_region = intensity_smooth[max(0, left_idx-20):left_idx]
        right_baseline_region = intensity_smooth[right_idx:min(len(intensity_smooth), right_idx+20)]
        
        if len(left_baseline_region) > 0 and len(right_baseline_region) > 0:
            baseline = (np.min(left_baseline_region) + np.min(right_baseline_region)) / 2
        elif len(left_baseline_region) > 0:
            baseline = np.min(left_baseline_region)
        elif len(right_baseline_region) > 0:
            baseline = np.min(right_baseline_region)
        else:
            baseline = baseline_mean
        
        # 计算净峰高（峰高减去基线）
        net_height = peak_height - baseline
        
        # 计算信噪比
        snr = net_height / baseline_noise
        
        # 计算峰面积（简单的梯形积分）
        peak_region = intensity_smooth[left_idx:right_idx+1]
        peak_area = np.trapz(peak_region - baseline, time[left_idx:right_idx+1])
        
        # 更严格的峰筛选条件
        conditions = [
            snr > snr_threshold,  # 信噪比要求
            net_height > max_intensity * 0.001,  # 相对高度要求
            peak_area > 0,  # 峰面积为正
            right_idx - left_idx >= 3  # 最小峰宽要求
        ]
        
        if all(conditions):
            peak_info = {
                'peak_number': len(peaks_info) + 1,
                'time': peak_time,
                'height': net_height,
                'start_time': start_time,
                'end_time': end_time,
                'snr': snr,
                'baseline': baseline,
                'area': peak_area,
                'width': end_time - start_time
            }
            peaks_info.append(peak_info)
    
    # 按峰面积排序，去除可能的假峰
    if len(peaks_info) > 1:
        peaks_info.sort(key=lambda x: x['area'], reverse=True)
        
        # 重新编号
        for i, peak in enumerate(peaks_info):
            peak['peak_number'] = i + 1
    
    return peaks_info, time, intensity, peaks

def print_peaks_table(peaks_info):
    """
    打印峰信息表格
    """
    print("\n=== 峰分析结果 ===")
    print(f"{'序号':<4} {'时间(min)':<10} {'净高度':<12} {'峰区间(min)':<20} {'信噪比':<8} {'面积':<12} {'峰宽':<8}")
    print("-" * 80)
    
    for peak in peaks_info:
        print(f"{peak['peak_number']:<4} {peak['time']:<10.4f} {peak['height']:<12.4f} "
              f"{peak['start_time']:.4f}-{peak['end_time']:.4f}  {peak['snr']:<8.2f} "
              f"{peak['area']:<12.4f} {peak['width']:<8.4f}")

def plot_peaks(time, intensity, peaks_info, peaks, intensity_smooth=None):
    """
    绘制色谱图和检测到的峰
    """
    plt.figure(figsize=(15, 8))
    
    # 绘制原始数据和平滑数据
    plt.subplot(2, 1, 1)
    plt.plot(time, intensity, 'lightblue', linewidth=1, label='原始数据', alpha=0.7)
    if intensity_smooth is not None:
        plt.plot(time, intensity_smooth, 'b-', linewidth=1, label='平滑数据')
    
    # 标记检测到的峰
    for peak in peaks_info:
        peak_idx = np.argmin(np.abs(time - peak['time']))
        if intensity_smooth is not None:
            peak_height = intensity_smooth[peak_idx]
        else:
            peak_height = intensity[peak_idx]
        plt.plot(peak['time'], peak_height, 'ro', markersize=8)
        plt.text(peak['time'], peak_height + max(intensity) * 0.02, 
                f"峰{peak['peak_number']}\nSNR:{peak['snr']:.1f}", 
                ha='center', va='bottom', fontsize=8)
    
    plt.xlabel('时间 (分钟)')
    plt.ylabel('强度')
    plt.title('色谱峰分析结果 - 全图')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # 绘制局部放大图（前10分钟）
    plt.subplot(2, 1, 2)
    mask = time <= 10
    plt.plot(time[mask], intensity[mask], 'lightblue', linewidth=1, label='原始数据', alpha=0.7)
    if intensity_smooth is not None:
        plt.plot(time[mask], intensity_smooth[mask], 'b-', linewidth=1, label='平滑数据')
    
    # 标记前10分钟的峰
    for peak in peaks_info:
        if peak['time'] <= 10:
            peak_idx = np.argmin(np.abs(time - peak['time']))
            if intensity_smooth is not None:
                peak_height = intensity_smooth[peak_idx]
            else:
                peak_height = intensity[peak_idx]
            plt.plot(peak['time'], peak_height, 'ro', markersize=8)
            plt.text(peak['time'], peak_height + max(intensity[mask]) * 0.05, 
                    f"峰{peak['peak_number']}", 
                    ha='center', va='bottom', fontsize=8)
    
    plt.xlabel('时间 (分钟)')
    plt.ylabel('强度')
    plt.title('色谱峰分析结果 - 前10分钟放大')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.show()

# 主程序
if __name__ == "__main__":
    # 获取脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 构建数据文件的完整路径
    file_path = os.path.join(script_dir, "AS301-24-10-01.dx_FID1A.txt")
    
    # 检查文件是否存在
    if not os.path.exists(file_path):
        print(f"错误：找不到数据文件 {file_path}")
        print(f"请确保文件 'AS301-24-10-01.dx_FID1A.txt' 在脚本同一目录下")
        exit(1)
    
    # 分析峰，使用更严格的参数
    peaks_info, time, intensity, all_peaks = analyze_peaks(file_path, 
                                                          snr_threshold=15,  # 提高信噪比阈值
                                                          min_peak_height_ratio=0.005)  # 最小峰高为最大值的0.5%
    
    # 打印结果
    print_peaks_table(peaks_info)
    
    # 重新读取数据进行绘图
    data = pd.read_csv(file_path, sep='\t', header=None, names=['time', 'intensity'])
    intensity_smooth = savgol_filter(data['intensity'].values, window_length=11, polyorder=3)
    
    # 绘制图形
    plot_peaks(time, intensity, peaks_info, all_peaks, intensity_smooth)
    
    # 保存结果到CSV文件
    if peaks_info:
        output_file = os.path.join(script_dir, 'peak_analysis_results_improved.csv')
        df_results = pd.DataFrame(peaks_info)
        df_results.to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"\n改进后的结果已保存到 {output_file}")
        print(f"共找到 {len(peaks_info)} 个符合条件的峰")
    else:
        print("\n未找到符合条件的峰")