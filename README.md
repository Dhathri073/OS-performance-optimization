# OS Performance Optimization Project

This project implements a machine learning-driven approach to optimize operating system performance using database management systems and algorithmic techniques.

## Features

- **Data Collection**: Collect OS performance metrics (CPU, memory, disk, network)
- **Database Storage**: Store historical performance data in SQLite database
- **Machine Learning Models**: Predict performance bottlenecks and optimize parameters
- **Algorithmic Optimization**: Use genetic algorithms and other techniques for parameter tuning

## Installation

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Run the main script: `python main.py`

## Usage

Use the virtual environment Python or the provided batch file:

- Collect data: `run.bat --collect --duration 60` or `"C:/Users/Dhathri M/OneDrive/Desktop/OSPerformanceoptimaization/.venv/Scripts/python.exe" main.py --collect --duration 60`
- Train models: `run.bat --train` or `"C:/Users/Dhathri M/OneDrive/Desktop/OSPerformanceoptimaization/.venv/Scripts/python.exe" main.py --train`
- Optimize: `run.bat --optimize` or `"C:/Users/Dhathri M/OneDrive/Desktop/OSPerformanceoptimaization/.venv/Scripts/python.exe" main.py --optimize`

**Add `--plot` flag to any command to display graphical results:**
- `run.bat --collect --plot --duration 30` - Shows performance metrics graphs
- `run.bat --train --plot` - Shows model performance plots
- **Start Web Interface**: `run.bat --web` or `"C:/Users/Dhathri M/OneDrive/Desktop/OSPerformanceoptimaization/.venv/Scripts/python.exe" main.py --web`

## Web Interface

The web interface provides:
- **Dashboard**: Real-time performance metrics with interactive charts
- **Data Collection**: Web form to collect performance data
- **ML Training**: One-click model training
- **Optimization**: Run genetic algorithm optimization
- **Visualization**: Plotly-powered interactive graphs

To start the web server:
```
C:\Users\Dhathri M\OneDrive\Desktop\OSPerformanceoptimaization\templates\index.html
```
Then open http://localhost:5000 in your browser.

## Launch Instructions

To run the project:

1. Ensure Python environment is activated: The project uses a virtual environment at `.venv/`
2. Use the following commands:
   - Data collection: `python main.py --collect` (but use the full venv path)
   - Model training: `python main.py --train` (but use the full venv path)
   - Optimization: `python main.py --optimize` (but use the full venv path)

Or simply use the `run.bat` file for convenience.

## Requirements

- Python 3.8+
- Libraries: psutil, scikit-learn, sqlite3, deap, etc.