"""
Data Collector Module
Collects OS performance metrics using psutil.
"""

import time
import psutil
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

def collect_data(duration=60, interval=1, plot=False):
    """
    Collect OS performance data for a given duration.

    Args:
        duration (int): Collection duration in seconds
        interval (int): Interval between measurements in seconds
        plot (bool): Whether to display plots of the collected data

    Returns:
        pd.DataFrame: Collected performance data
    """
    data = []
    start_time = time.time()

    print(f"Collecting OS performance data for {duration} seconds...")
    while time.time() - start_time < duration:
        timestamp = datetime.now()

        # CPU metrics
        cpu_percent = psutil.cpu_percent(interval=None)
        cpu_freq = psutil.cpu_freq().current if psutil.cpu_freq() else 0

        # Memory metrics
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        memory_used = memory.used / (1024**3)  # GB

        # Disk metrics
        disk = psutil.disk_usage('/')
        disk_percent = disk.percent
        disk_used = disk.used / (1024**3)  # GB

        # Network metrics
        net = psutil.net_io_counters()
        bytes_sent = net.bytes_sent / (1024**2)  # MB
        bytes_recv = net.bytes_recv / (1024**2)  # MB

        data.append({
            'timestamp': timestamp,
            'cpu_percent': cpu_percent,
            'cpu_freq': cpu_freq,
            'memory_percent': memory_percent,
            'memory_used_gb': memory_used,
            'disk_percent': disk_percent,
            'disk_used_gb': disk_used,
            'net_sent_mb': bytes_sent,
            'net_recv_mb': bytes_recv
        })

        time.sleep(interval)

    df = pd.DataFrame(data)
    print("Data collection complete.")

    if plot and not df.empty:
        plot_performance_data(df)

    return df

def plot_performance_data(df):
    """
    Plot the collected performance data.

    Args:
        df (pd.DataFrame): Performance data
    """
    # Set up the plotting style
    sns.set_style("darkgrid")
    plt.style.use('seaborn-v0_8')

    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('OS Performance Metrics Over Time', fontsize=16, fontweight='bold')

    # Convert timestamp to relative time for better plotting
    df['relative_time'] = (df['timestamp'] - df['timestamp'].iloc[0]).dt.total_seconds()

    # CPU Usage
    axes[0, 0].plot(df['relative_time'], df['cpu_percent'], 'b-', linewidth=2, marker='o', markersize=3)
    axes[0, 0].set_title('CPU Usage (%)', fontweight='bold')
    axes[0, 0].set_xlabel('Time (seconds)')
    axes[0, 0].set_ylabel('CPU %')
    axes[0, 0].grid(True, alpha=0.3)

    # Memory Usage
    axes[0, 1].plot(df['relative_time'], df['memory_percent'], 'r-', linewidth=2, marker='s', markersize=3)
    axes[0, 1].set_title('Memory Usage (%)', fontweight='bold')
    axes[0, 1].set_xlabel('Time (seconds)')
    axes[0, 1].set_ylabel('Memory %')
    axes[0, 1].grid(True, alpha=0.3)

    # Disk Usage
    axes[1, 0].plot(df['relative_time'], df['disk_percent'], 'g-', linewidth=2, marker='^', markersize=3)
    axes[1, 0].set_title('Disk Usage (%)', fontweight='bold')
    axes[1, 0].set_xlabel('Time (seconds)')
    axes[1, 0].set_ylabel('Disk %')
    axes[1, 0].grid(True, alpha=0.3)

    # Network I/O
    ax2 = axes[1, 1].twinx()
    line1 = axes[1, 1].plot(df['relative_time'], df['net_sent_mb'], 'c-', linewidth=2, marker='*', markersize=3, label='Sent')
    line2 = ax2.plot(df['relative_time'], df['net_recv_mb'], 'm-', linewidth=2, marker='D', markersize=3, label='Received')
    axes[1, 1].set_title('Network I/O (MB)', fontweight='bold')
    axes[1, 1].set_xlabel('Time (seconds)')
    axes[1, 1].set_ylabel('Sent (MB)', color='c')
    ax2.set_ylabel('Received (MB)', color='m')
    axes[1, 1].grid(True, alpha=0.3)

    # Combine legends
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    axes[1, 1].legend(lines, labels, loc='upper right')

    plt.tight_layout()
    plt.savefig('performance_metrics.png', dpi=300, bbox_inches='tight')
    print("Performance metrics plot saved as 'performance_metrics.png'")
    plt.close()  # Close the figure to free memory

if __name__ == '__main__':
    df = collect_data(duration=10, plot=True)
    print(df.head())