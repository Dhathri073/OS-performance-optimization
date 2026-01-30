"""
Web Application Module
Provides a web interface for the OS Performance Optimization tool.
"""

from flask import Flask, render_template, request, jsonify, redirect, url_for
import json
import pandas as pd
from datetime import datetime
import plotly
import plotly.graph_objs as go
from plotly.utils import PlotlyJSONEncoder
import os

from src.collectors.data_collector import collect_data
from src.database.db_manager import init_db, store_data, load_data
from src.ml.model_trainer import train_model, predict_performance
from src.algorithms.optimizer import optimize_parameters

# Configure Flask app with correct template and static folders
template_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'templates')
static_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'static')
app = Flask(__name__, template_folder=template_dir, static_folder=static_dir)

@app.route('/')
def index():
    """Main dashboard page."""
    return render_template('index.html')

@app.route('/collect', methods=['GET', 'POST'])
def collect():
    """Data collection page."""
    if request.method == 'POST':
        try:
            duration = int(request.form.get('duration', 10))
            data = collect_data(duration=duration)
            init_db()
            store_data(data)
            return redirect(url_for('dashboard'))
        except Exception as e:
            # Return to collect page with error message
            return render_template('collect.html', error=str(e))
    return render_template('collect.html')

@app.route('/dashboard')
def dashboard():
    """Dashboard with data visualization."""
    try:
        df = load_data(limit=1000)

        if df.empty:
            return render_template('dashboard.html', data=None, graphs=None)

        # Create graphs
        graphs = create_performance_graphs(df)

        # Get latest metrics
        latest = df.iloc[-1] if not df.empty else None
        if latest is not None:
            latest = latest.to_dict()  # Convert Series to dict for template access

        return render_template('dashboard.html', data=latest, graphs=graphs)
    except Exception as e:
        return render_template('dashboard.html', data=None, graphs=None, error=str(e))

@app.route('/train', methods=['POST'])
def train():
    """Train the ML model."""
    try:
        result = train_model()
        return jsonify({'status': 'success', 'message': 'Model trained successfully'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/optimize', methods=['POST'])
def optimize():
    """Run optimization."""
    try:
        result = optimize_parameters()
        return jsonify({
            'status': 'success',
            'parameters': result,
            'message': f'Optimization complete. Best fitness: {result.fitness.values[0]:.4f}'
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/api/data')
def get_data():
    """API endpoint to get performance data."""
    df = load_data(limit=100)
    return jsonify(df.to_dict('records'))

def create_performance_graphs(df):
    """Create Plotly graphs for performance metrics."""
    graphs = []

    # CPU Usage Graph
    cpu_graph = {
        'data': [go.Scatter(
            x=df['timestamp'],
            y=df['cpu_percent'],
            mode='lines',
            name='CPU Usage (%)'
        )],
        'layout': go.Layout(
            title='CPU Usage Over Time',
            xaxis={'title': 'Time'},
            yaxis={'title': 'CPU %'}
        )
    }
    graphs.append(cpu_graph)

    # Memory Usage Graph
    memory_graph = {
        'data': [go.Scatter(
            x=df['timestamp'],
            y=df['memory_percent'],
            mode='lines',
            name='Memory Usage (%)'
        )],
        'layout': go.Layout(
            title='Memory Usage Over Time',
            xaxis={'title': 'Time'},
            yaxis={'title': 'Memory %'}
        )
    }
    graphs.append(memory_graph)

    # Disk Usage Graph
    disk_graph = {
        'data': [go.Scatter(
            x=df['timestamp'],
            y=df['disk_percent'],
            mode='lines',
            name='Disk Usage (%)'
        )],
        'layout': go.Layout(
            title='Disk Usage Over Time',
            xaxis={'title': 'Time'},
            yaxis={'title': 'Disk %'}
        )
    }
    graphs.append(disk_graph)

    # Convert to JSON
    graphs_json = []
    for graph in graphs:
        graphs_json.append(json.dumps(graph, cls=PlotlyJSONEncoder))

    return graphs_json

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)