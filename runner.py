import logging
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL
logging.getLogger('tensorflow').setLevel(logging.FATAL)
import numpy as np
import argparse
import pandas as pd
import random

# export KMP_DUPLICATE_LIB_OK=TRUE

def parse_args():
    parser = argparse.ArgumentParser(
                        description='Variational Combinatorial Sequential Monte Carlo')
    parser.add_argument('--dataset',
                        help='benchmark dataset to use.',
                        default='primate_data')
    parser.add_argument('--n_particles',
                        type=int,
                        help='number of SMC samples.',
                        default=128)
    parser.add_argument('--batch_size',
                        type=int,
                        help='number of sites on genome per batch.',
                        default=256)
    parser.add_argument('--learning_rate',
                        type=float,
                        help='Learning rate.',
                        default=0.011)
    parser.add_argument('--num_epoch',
                        type=int,
                        help='number of epoches to train.',
                        default=1000)
    parser.add_argument('--optimizer',
                       type=str,
                       help='Optimizer for Training',
                       default='GradientDescentOptimizer')
    parser.add_argument('--branch_prior',
                       type=float,
                       help='Hyperparameter for branch length initialization.',
                       default=np.log(5))
    parser.add_argument('--lambda_prior',
                       type=float,
                       help='Hyperparameter for branch length initialization.',
                       default=3)
    parser.add_argument('--decay_prior',
                       type=float,
                       help='Hyperparameter for branch length initialization.',
                       default=np.log(5))
    parser.add_argument('--M',
                       type=int,
                       help='number of subparticles to compute look-ahead particles',
                       default=10)
    parser.add_argument('--nested', 
                       default=False, 
                       type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--jcmodel', 
                       default=False, 
                       type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--memory_optimization',
                       help='Use memory optimization?',
                       default='on')


    args = parser.parse_args()
    return args


if __name__ == "__main__":

    primate_data = False
    corona_data = False
    hohna_data = False
    load_strings = False
    simulate_data = False
    hohna_data_1 = False
    hohna_data_2 = False
    hohna_data_3 = False
    hohna_data_4 = False
    hohna_data_5 = False
    hohna_data_6 = False
    hohna_data_7 = False
    hohna_data_8 = False
    primate_data_wang = False

    ginkgo = True

    args = parse_args()

    exec(args.dataset + ' = True')

    Alphabet_dir = {'A': [1, 0, 0, 0],
                    'C': [0, 1, 0, 0],
                    'G': [0, 0, 1, 0],
                    'T': [0, 0, 0, 1]}
    alphabet_dir = {'a': [1, 0, 0, 0],
                    'c': [0, 1, 0, 0],
                    'g': [0, 0, 1, 0],
                    't': [0, 0, 0, 1]}
    Alphabet_dir_blank = {'A': [1, 0, 0, 0],
                          'C': [0, 1, 0, 0],
                          'G': [0, 0, 1, 0],
                          'T': [0, 0, 0, 1],
                          '-': [1, 1, 1, 1],
                          '?': [1, 1, 1, 1]}
    alphabet = np.array([[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 1.]])


    def simulateDNA(nsamples, seqlength, alphabet):
        genomes_NxSxA = np.zeros([nsamples, seqlength, alphabet.shape[0]])
        for n in range(nsamples):
            genomes_NxSxA[n] = np.array([random.choice(alphabet) for i in range(seqlength)])
        return genomes_NxSxA


    def form_dataset_from_strings(genome_strings, alphabet_dir, alphabet_num=4):
        genomes_NxSxA = np.zeros([len(genome_strings), len(genome_strings[0]), alphabet_num])
        for i in range(genomes_NxSxA.shape[0]):
            for j in range(genomes_NxSxA.shape[1]):
                genomes_NxSxA[i, j] = alphabet_dir[genome_strings[i][j]]
        taxa = ['S' + str(i) for i in range(genomes_NxSxA.shape[0])]
        datadict = {'taxa': taxa,
                    'genome': genomes_NxSxA}
        return datadict

    if hohna_data or hohna_data_1:
        datadict_raw = pd.read_pickle('data/hohna_datasets/DS1.pickle')
        genome_strings = list(datadict_raw.values())
        datadict = form_dataset_from_strings(genome_strings, Alphabet_dir_blank)
        # print(datadict['genome'].shape)
        
    if hohna_data_2:
        datadict_raw = pd.read_pickle('data/hohna_datasets/DS2.pickle')
        genome_strings = list(datadict_raw.values())
        datadict = form_dataset_from_strings(genome_strings, Alphabet_dir_blank)
        
    if hohna_data_3:
        datadict_raw = pd.read_pickle('data/hohna_datasets/DS3.pickle')
        genome_strings = list(datadict_raw.values())
        datadict = form_dataset_from_strings(genome_strings, Alphabet_dir_blank)
        
    if hohna_data_4:
        datadict_raw = pd.read_pickle('data/hohna_datasets/DS4.pickle')
        genome_strings = list(datadict_raw.values())
        datadict = form_dataset_from_strings(genome_strings, Alphabet_dir_blank)
        
    if hohna_data_5:
        datadict_raw = pd.read_pickle('data/hohna_datasets/DS5.pickle')
        genome_strings = list(datadict_raw.values())
        datadict = form_dataset_from_strings(genome_strings, Alphabet_dir_blank)
        
    if hohna_data_6:
        datadict_raw = pd.read_pickle('data/hohna_datasets/DS6.pickle')
        genome_strings = list(datadict_raw.values())
        datadict = form_dataset_from_strings(genome_strings, Alphabet_dir_blank)
        
    if hohna_data_7:
        datadict_raw = pd.read_pickle('data/hohna_datasets/DS7.pickle')
        genome_strings = list(datadict_raw.values())
        datadict = form_dataset_from_strings(genome_strings, Alphabet_dir_blank)
        
    if hohna_data_8:
        datadict_raw = pd.read_pickle('data/hohna_datasets/DS8.pickle')
        genome_strings = list(datadict_raw.values())
        datadict = form_dataset_from_strings(genome_strings, Alphabet_dir_blank)


    if corona_data:
        datadict = pd.read_pickle('data/coronavirus.p')


    if primate_data:
        datadict_raw = pd.read_pickle('data/primate.p')
        genome_strings = list(datadict_raw.values())
        datadict = form_dataset_from_strings(genome_strings, Alphabet_dir_blank)
        
    if primate_data_wang:
        datadict_raw = pd.read_pickle('data/primates_small.p')
        genome_strings = list(datadict_raw.values())
        datadict = form_dataset_from_strings(genome_strings, Alphabet_dir)


    if simulate_data:
        data_NxSxA = simulateDNA(3, 5, alphabet)
        # print("Simulated genomes:\n", data_NxSxA)
        taxa = ['S' + str(i) for i in range(data_NxSxA.shape[0])]
        datadict = {'taxa': taxa,
                    'genome': data_NxSxA}


    if load_strings:
        genome_strings = ['ACTTTGAGAG', 'ACTTTGACAG', 'ACTTTGACTG', 'ACTTTGACTC']
        datadict = form_dataset_from_strings(genome_strings, Alphabet_dir)

    if ginkgo:
        all_datadict = pd.read_pickle('data/gingko/10_2p5_jets.p')
        # print(datadict)
        # print(datadict)
        leafNums = []
        decay_params = []
        for i in range(1,2):
            leafs = len(all_datadict[i]["data"])
            print("number of leaves:", leafs)
            if leafs <= 10:
                continue
            leafNums.append(leafs)
            jetData = np.expand_dims(all_datadict[i]["data"], axis = 1)
            llhsum = all_datadict[i]['llh']
            # print(jetData.shape)
            datadict = {}
            datadict['data'] = jetData
            datadict['samples'] = [str(i) for i in range(len(datadict['data']))]
            datadict['llh'] = llhsum
            print("ground truth llh", datadict['llh'])
        # print(datadict)
        #import pdb
        # pdb.set_trace()


            if args.nested == True:
                import vncsmc as vcsmc
            elif not ginkgo:
                import vcsmc as vcsmc
            elif ginkgo:
                import jet_vcsmc as vcsmc


            #pdb.set_trace()
            vcsmc = vcsmc.VCSMC(datadict, K=args.n_particles, args=args)

            param = vcsmc.train(epochs=args.num_epoch, batch_size=args.batch_size, learning_rate=args.learning_rate, memory_optimization=args.memory_optimization)
            
            decay_params.append(param)
        print(decay_params)
        print("final estimate: " + str(np.average(decay_params)))
        print(leafNums)
    else:    
        if args.nested == True:
            import vncsmc as vcsmc
        elif not ginkgo:
            import vcsmc as vcsmc
        elif ginkgo:
            import jet_vcsmc as vcsmc


        #pdb.set_trace()
        vcsmc = vcsmc.VCSMC(datadict, K=args.n_particles, args=args)

        param = vcsmc.train(epochs=args.num_epoch, batch_size=args.batch_size, learning_rate=args.learning_rate, memory_optimization=args.memory_optimization)