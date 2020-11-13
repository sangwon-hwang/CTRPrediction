## Required Functions Imported and MockGenerator Defined

from molvae.utils.mgen import MoleculeGenerator

import torch
from typing import List
from guacamol.goal_directed_benchmark import GoalDirectedBenchmark
from guacamol.scoring_function import ArithmeticMeanScoringFunction
from guacamol.standard_benchmarks import hard_cobimetinib, similarity, logP_benchmark, cns_mpo, \
    qed_benchmark, median_camphor_menthol, novelty_benchmark, isomers_c11h24, isomers_c7h8n2o2, isomers_c9h10n2o2pf2cl, \
    frechet_benchmark, tpsa_benchmark, hard_osimertinib, hard_fexofenadine, weird_physchem, start_pop_ranolazine, \
    kldiv_benchmark, perindopril_rings, amlodipine_rings, sitagliptin_replacement, zaleplon_with_other_formula, valsartan_smarts, \
    median_tadalafil_sildenafil, decoration_hop, scaffold_hop, ranolazine_mpo, pioglitazone_mpo
from guacamol.distribution_learning_benchmark import DistributionLearningBenchmark, ValidityBenchmark, \
    UniquenessBenchmark
from rdkit import Chem
import torch
import random
import sys


def goal_directed_benchmark_suite(version_name: str) -> List[GoalDirectedBenchmark]:
    if version_name == 'v1':
        return goal_directed_suite_v1()
    if version_name == 'v2':
        return goal_directed_suite_v2()
    if version_name == 'trivial':
        return goal_directed_suite_trivial()

    raise Exception(f'Goal-directed benchmark suite "{version_name}" does not exist.')


def distribution_learning_benchmark_suite(chembl_file_path: str,
                                          version_name: str,
                                          number_samples: int) -> List[DistributionLearningBenchmark]:
    """
    Returns a suite of benchmarks for a specified benchmark version
    Args:
        chembl_file_path: path to ChEMBL training set, necessary for some benchmarks
        version_name: benchmark version
    Returns:
        List of benchmaks
    """

    # For distribution-learning, v1 and v2 are identical
    if version_name == 'v1' or version_name == 'v2':
        return distribution_learning_suite_v1(chembl_file_path=chembl_file_path, number_samples=number_samples)

    raise Exception(f'Distribution-learning benchmark suite "{version_name}" does not exist.')


def goal_directed_suite_v1() -> List[GoalDirectedBenchmark]:
    max_logP = 6.35584
    return [
        isomers_c11h24(mean_function='arithmetic'),
        isomers_c7h8n2o2(mean_function='arithmetic'),
        isomers_c9h10n2o2pf2cl(mean_function='arithmetic', n_samples=100),

        hard_cobimetinib(max_logP=max_logP),
        hard_osimertinib(ArithmeticMeanScoringFunction),
        hard_fexofenadine(ArithmeticMeanScoringFunction),
        weird_physchem(),

        # start pop benchmark
        # e.g.
        start_pop_ranolazine(),

        # similarity Benchmarks

        # explicit rediscovery
        similarity(smiles='CC1=CC=C(C=C1)C1=CC(=NN1C1=CC=C(C=C1)S(N)(=O)=O)C(F)(F)F', name='Celecoxib',
                   fp_type='ECFP4', threshold=1.0, rediscovery=True),
        similarity(smiles='Cc1c(C)c2OC(C)(COc3ccc(CC4SC(=O)NC4=O)cc3)CCc2c(C)c1O', name='Troglitazone',
                   fp_type='ECFP4', threshold=1.0, rediscovery=True),
        similarity(smiles='CN(C)S(=O)(=O)c1ccc2Sc3ccccc3C(=CCCN4CCN(C)CC4)c2c1', name='Thiothixene',
                   fp_type='ECFP4', threshold=1.0, rediscovery=True),

        # generate similar stuff
        similarity(smiles='Clc4cccc(N3CCN(CCCCOc2ccc1c(NC(=O)CC1)c2)CC3)c4Cl',
                   name='Aripiprazole', fp_type='FCFP4', threshold=0.75),
        similarity(smiles='CC(C)(C)NCC(O)c1ccc(O)c(CO)c1', name='Albuterol',
                   fp_type='FCFP4', threshold=0.75),
        similarity(smiles='COc1ccc2[C@H]3CC[C@@]4(C)[C@@H](CC[C@@]4(O)C#C)[C@@H]3CCc2c1', name='Mestranol',
                   fp_type='AP', threshold=0.75),

        logP_benchmark(target=-1.0),
        logP_benchmark(target=8.0),
        tpsa_benchmark(target=150.0),

        cns_mpo(max_logP=max_logP),
        qed_benchmark(),
        median_camphor_menthol(ArithmeticMeanScoringFunction)
    ]


def goal_directed_suite_v2() -> List[GoalDirectedBenchmark]:
    return [
        # explicit rediscovery
        similarity(smiles='CC1=CC=C(C=C1)C1=CC(=NN1C1=CC=C(C=C1)S(N)(=O)=O)C(F)(F)F', name='Celecoxib', fp_type='ECFP4',
                   threshold=1.0, rediscovery=True),
        similarity(smiles='Cc1c(C)c2OC(C)(COc3ccc(CC4SC(=O)NC4=O)cc3)CCc2c(C)c1O', name='Troglitazone', fp_type='ECFP4',
                   threshold=1.0, rediscovery=True),
        similarity(smiles='CN(C)S(=O)(=O)c1ccc2Sc3ccccc3C(=CCCN4CCN(C)CC4)c2c1', name='Thiothixene', fp_type='ECFP4',
                   threshold=1.0, rediscovery=True),

        # generate similar stuff
        similarity(smiles='Clc4cccc(N3CCN(CCCCOc2ccc1c(NC(=O)CC1)c2)CC3)c4Cl', name='Aripiprazole', fp_type='ECFP4',
                   threshold=0.75),
        similarity(smiles='CC(C)(C)NCC(O)c1ccc(O)c(CO)c1', name='Albuterol', fp_type='FCFP4', threshold=0.75),
        similarity(smiles='COc1ccc2[C@H]3CC[C@@]4(C)[C@@H](CC[C@@]4(O)C#C)[C@@H]3CCc2c1', name='Mestranol',
                   fp_type='AP', threshold=0.75),

        # isomers
        isomers_c11h24(),
        isomers_c9h10n2o2pf2cl(),

        # median molecules
        median_camphor_menthol(),
        median_tadalafil_sildenafil(),

        # all other MPOs
        hard_osimertinib(),
        hard_fexofenadine(),
        ranolazine_mpo(),
        perindopril_rings(),
        amlodipine_rings(),
        sitagliptin_replacement(),
        zaleplon_with_other_formula(),
        valsartan_smarts(),
        decoration_hop(),
        scaffold_hop(),
    ]


def goal_directed_suite_trivial() -> List[GoalDirectedBenchmark]:
    """
    Trivial goal-directed benchmarks from the paper.
    """
    return [
        logP_benchmark(target=-1.0),
        logP_benchmark(target=8.0),
        tpsa_benchmark(target=150.0),
        cns_mpo(),
        qed_benchmark(),
        isomers_c7h8n2o2(),
        pioglitazone_mpo(),
    ]


def distribution_learning_suite_v1(chembl_file_path: str, number_samples: int = 10000) -> \
        List[DistributionLearningBenchmark]:
    """
    Suite of distribution learning benchmarks, v1.
    Args:
        chembl_file_path: path to the file with the reference ChEMBL molecules
    Returns:
        List of benchmarks, version 1
    """
    return [
        ValidityBenchmark(number_samples=number_samples),
        UniquenessBenchmark(number_samples=number_samples),
        novelty_benchmark(training_set_file=chembl_file_path, number_samples=number_samples),
        kldiv_benchmark(training_set_file=chembl_file_path, number_samples=number_samples),
        frechet_benchmark(training_set_file=chembl_file_path, number_samples=number_samples)
    ]

# Convey method : MockGenerator

from typing import List
from guacamol.distribution_matching_generator import DistributionMatchingGenerator

class MockGenerator(DistributionMatchingGenerator):
    """
    Mock generator that returns pre-defined molecules,
    possibly split in several calls
    """

    def __init__(self, molecules: List[str]) -> None:    # static typing in Python 3.6
        self.molecules = molecules                      
        self.cursor = 0

    def generate(self, number_samples: int) -> List[str]:
        end = self.cursor + number_samples

        sampled_molecules = self.molecules[self.cursor:end]  # self.molecules list type
        self.cursor = end
        
        return sampled_molecules

if __name__ == '__main__':
    trainig_data_path = sys.argv[1]
    valid_data_path   = sys.argv[2]

    # Text load 
    population = []
    genList = []

    with open(trainig_data_path, 'r') as file:
        for text in file:
            population.append(text.strip('\n'))

    with open(valid_data_path, 'r') as file:
        for text in file:
            genList.append(text.strip('\n'))

    print(len(genList))
    print(len(population))    

    if len(population) < len(genList):
        sample_size = len(population) 
        genList = random.sample(genList, sample_size)
    else:
        sample_size = len(genList)
        population = random.sample(population, sample_size)
        
    result = distribution_learning_suite_v1(chembl_file_path=trainig_data_path, number_samples=sample_size)

    mock = MockGenerator(genList)
    validityBenchmark = result[0].assess_model(mock)
    validity = {
        'benchmark': validityBenchmark.benchmark_name,
        'metadata' : validityBenchmark.metadata,
        'score'  : validityBenchmark.score
    }

    mock = MockGenerator(genList)
    uniquenessBenchmark = result[1].assess_model(mock)
    Uniqueness = {
        'benchmark': uniquenessBenchmark.benchmark_name,
        'metadata': uniquenessBenchmark.metadata,
        'score'  : uniquenessBenchmark.score
    }

    mock = MockGenerator(genList)
    noveltyBenchmark = result[2].assess_model(mock)
    Novelty = {
        'benchmark': noveltyBenchmark.benchmark_name,
        'metadata': noveltyBenchmark.metadata,
        'score'  : noveltyBenchmark.score
    }

    mock = MockGenerator(genList)
    kLDivBenchmark = result[3].assess_model(mock)
    kLDiv = {
        'benchmark': kLDivBenchmark.benchmark_name,
        'metadata': kLDivBenchmark.metadata,
        'score'  : kLDivBenchmark.score
    }

    mock = MockGenerator(genList)
    FCDBenchmark = result[4].assess_model(mock)
    FCD = {
        'benchmark': FCDBenchmark.benchmark_name,
        'metadata': FCDBenchmark.metadata,
        'score'  : FCDBenchmark.score
    }
   
    print(validity, Uniqueness, Novelty, kLDiv, FCD)



