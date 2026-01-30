"""
Optimization Algorithms Module
Uses genetic algorithms to optimize OS parameters.
"""

import random
import matplotlib.pyplot as plt
import seaborn as sns
from deap import base, creator, tools, algorithms
import numpy as np
from src.ml.model_trainer import predict_performance

# Define optimization problem
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

def evaluate_individual(individual):
    """
    Evaluate an individual (parameter set) by predicting performance.

    Args:
        individual (list): Parameter values

    Returns:
        tuple: Fitness value
    """
    # Assume parameters are: [cpu_priority, memory_limit, disk_cache_size]
    # Map to features for prediction
    features = {
        'cpu_percent': individual[0] * 100,  # Scale to percent
        'cpu_freq': individual[1] * 3000,    # Scale to MHz
        'disk_percent': individual[2] * 100,
        'disk_used_gb': 50,  # Fixed for simplicity
        'net_sent_mb': 10,
        'net_recv_mb': 10
    }

    prediction = predict_performance(features)
    if prediction is None:
        return (0,)  # Default fitness

    # Higher memory usage might indicate better performance in some contexts
    # For demo, maximize predicted memory usage (inverse of optimization)
    return (prediction,)

def optimize_parameters(plot=False):
    """Run genetic algorithm optimization."""
    toolbox = base.Toolbox()

    # Define parameter ranges (normalized 0-1)
    toolbox.register("attr_float", random.uniform, 0, 1)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=3)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("evaluate", evaluate_individual)
    toolbox.register("mate", tools.cxBlend, alpha=0.5)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.1, indpb=0.2)
    toolbox.register("select", tools.selTournament, tournsize=3)

    # Create population
    population = toolbox.population(n=50)

    # Statistics
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    hof = tools.HallOfFame(1)

    # Run evolution
    print("Running optimization...")
    pop, log = algorithms.eaSimple(population, toolbox, cxpb=0.5, mutpb=0.2, ngen=10,
                                   stats=stats, halloffame=hof, verbose=True)

    # Best individual
    best = hof[0]
    print(f"Best parameters: {best}")
    print(f"Best fitness: {best.fitness.values[0]}")

    if plot:
        plot_optimization_progress(log)

    return best

def plot_optimization_progress(log):
    """
    Plot the optimization progress over generations.

    Args:
        log: DEAP logbook with statistics
    """
    sns.set_style("darkgrid")
    plt.style.use('seaborn-v0_8')

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Genetic Algorithm Optimization Progress', fontsize=16, fontweight='bold')

    gen = log.select("gen")
    avg = log.select("avg")
    std = log.select("std")
    min_ = log.select("min")
    max_ = log.select("max")

    # Fitness over generations
    axes[0, 0].plot(gen, max_, 'r-', linewidth=2, marker='o', markersize=4, label='Maximum')
    axes[0, 0].plot(gen, avg, 'b-', linewidth=2, marker='s', markersize=4, label='Average')
    axes[0, 0].fill_between(gen, np.array(avg)-np.array(std), np.array(avg)+np.array(std),
                           alpha=0.2, color='blue', label='±1 Std Dev')
    axes[0, 0].set_xlabel('Generation')
    axes[0, 0].set_ylabel('Fitness')
    axes[0, 0].set_title('Fitness Evolution', fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Best fitness over time
    axes[0, 1].plot(gen, max_, 'r-', linewidth=3, marker='o', markersize=6)
    axes[0, 1].set_xlabel('Generation')
    axes[0, 1].set_ylabel('Best Fitness')
    axes[0, 1].set_title('Best Fitness Progress', fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)

    # Average fitness
    axes[1, 0].plot(gen, avg, 'g-', linewidth=2, marker='^', markersize=4)
    axes[1, 0].set_xlabel('Generation')
    axes[1, 0].set_ylabel('Average Fitness')
    axes[1, 0].set_title('Average Fitness Trend', fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)

    # Fitness distribution (final generation)
    if len(max_) > 0:
        final_gen_data = [max_[-1], avg[-1], min_[-1]]
        labels = ['Max', 'Avg', 'Min']
        colors = ['red', 'blue', 'green']
        axes[1, 1].bar(labels, final_gen_data, color=colors, alpha=0.7, edgecolor='black')
        axes[1, 1].set_ylabel('Fitness Value')
        axes[1, 1].set_title('Final Generation Statistics', fontweight='bold')
        axes[1, 1].grid(True, alpha=0.3, axis='y')

        # Add value labels on bars
        for i, v in enumerate(final_gen_data):
            axes[1, 1].text(i, v + 0.01, f'{v:.2f}', ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    plt.savefig('optimization_progress.png', dpi=300, bbox_inches='tight')
    print("Optimization progress plot saved as 'optimization_progress.png'")
    plt.close()

if __name__ == '__main__':
    optimize_parameters(plot=True)