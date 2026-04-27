#!/usr/bin/env python3
"""
Machine Learning-Driven OS Performance Optimization
Main entry point for the application.
"""

import argparse
import sys
from src.collectors.data_collector import collect_data
from src.database.db_manager import init_db, store_data
from src.ml.model_trainer import train_model
from src.algorithms.optimizer import optimize_parameters

def main():
    parser = argparse.ArgumentParser(description='OS Performance Optimization Tool')
    parser.add_argument('--collect', action='store_true', help='Collect performance data')
    parser.add_argument('--train', action='store_true', help='Train ML model')
    parser.add_argument('--optimize', action='store_true', help='Run optimization')
    parser.add_argument('--web', action='store_true', help='Start web interface')
    parser.add_argument('--plot', action='store_true', help='Generate plots')
    parser.add_argument('--duration', type=int, default=60, help='Data collection duration in seconds')

    args = parser.parse_args()

    if args.web:
        print("Starting web interface...")
        from src.web_app import app
        app.run(debug=True, host='0.0.0.0', port=5000)
        return

    if args.collect:
        print("Collecting OS performance data...")
        data = collect_data(duration=args.duration, plot=args.plot)
        init_db()
        store_data(data)
        print("Data collection complete.")

    elif args.train:
        print("Training ML model...")
        train_model(plot=args.plot)
        print("Model training complete.")

    elif args.optimize:
        print("Running optimization...")
        optimize_parameters(plot=args.plot)
        print("Optimization complete.")

    else:
        print("Use --collect, --train, --optimize, or --web flags.")
        print("Add --plot flag to display graphical results.")
        print("Examples:")
        print("  python main.py --collect --duration 30")
        print("  python main.py --train")
        print("  python main.py --optimize")
        print("  python main.py --web")
        sys.exit(1)

if __name__ == '__main__':
    main()