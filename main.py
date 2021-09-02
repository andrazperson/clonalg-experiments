import numpy as np
import pandas as pd
from niapy.task import Task
from niapy.algorithms.basic import (
    ArtificialBeeColonyAlgorithm,
    BatAlgorithm,
    ClonalSelectionAlgorithm,
    CuckooSearch,
    DifferentialEvolution,
    ParticleSwarmAlgorithm,
)


def run_experiment(algorithms, problems, runs=5, dimension=10, max_iters=1000):
    functions = [problem[0] for problem in problems]
    iterables = [functions, ('Min', 'Max', 'Avg', 'Std')]
    index = pd.MultiIndex.from_product(iterables)
    columns = [algorithm.Name[1] for algorithm in algorithms]
    dataframe = pd.DataFrame(np.zeros((len(problems) * 4, len(algorithms))), index, columns)
    for problem in problems:
        name, lower, upper = problem
        for algorithm in algorithms:
            fitness = []
            for i in range(runs):
                task = Task(name, lower=lower, upper=upper, dimension=dimension, max_iters=max_iters)
                _, best_fitness = algorithm.run(task)
                fitness.append(best_fitness)
            dataframe[algorithm.Name[1]][name] = (np.min(fitness), np.max(fitness), np.mean(fitness), np.std(fitness))
    dataframe.to_pickle('output/result.pkl'.format(dimension))


if __name__ == '__main__':
    problems = (
        ('ackley', -32.768, 32.768),
        ('alpine1', -10, 10),
        ('discus', -100, 100),
        ('griewank', -600, 600),
        ('levy', -10, 10),
        ('pinter', -10, 10),
        ('salomon', -100, 100),
        ('sphere', -5.12, 5.12),
        ('whitley', -10.24, 10.24),
        ('zakharov', -5, 10)
    )

    abc = ArtificialBeeColonyAlgorithm(population_size=100, seed=12345)
    ba = BatAlgorithm(population_size=100, seed=12345)
    clonalg = ClonalSelectionAlgorithm(population_size=100, clone_factor=0.1,
                                       mutation_factor=-2.5, num_rand=1,
                                       bits_per_param=16, seed=12345)
    cs = CuckooSearch(population_size=100, seed=12345)
    de = DifferentialEvolution(population_size=100, differential_weight=0.8,
                               crossover_probability=0.9, seed=12345)
    pso = ParticleSwarmAlgorithm(population_size=100, w=0.9, c1=0.5, c2=0.3,
                                 min_velocity=-1, max_velocity=1, seed=12345)

    algorithms = (clonalg, abc, ba, cs, de, pso)
    run_experiment(algorithms, problems, 5, 10, 1000)
    pkl = pd.read_pickle('output/result.pkl')
    print(pkl)
