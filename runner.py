import vcsmc as vcsmc
import numpy as np
import argparse
import pandas as pd

# export KMP_DUPLICATE_LIB_OK=TRUE

def parse_args():
    parser = argparse.ArgumentParser(
        description='Variational Combinatorial Sequential Monte Carlo'
    )
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset',
        help='benchmark dataset to use.',
        default='load_strings')
    parser.add_argument('--memory_optimization',
                        help='Use memory optimization?',
                        default='on')
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
                        default=0.01)
    parser.add_argument(
        '--num_epoch',
        type=int,
        help='number of epoches to train.',
        default=100
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":

    primate_data = False
    corona_data = False
    hohna_data = False
    load_strings = False
    simulate_data = False


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
    Alphabet_dir_blank_old = {'A': [1, 0, 0, 0, 0],
                              'C': [0, 1, 0, 0, 0],
                              'G': [0, 0, 1, 0, 0],
                              'T': [0, 0, 0, 1, 0],
                              '-': [0, 0, 0, 0, 1]}
    Alphabet_dir_blank = {'A': [1, 0, 0, 0],
                          'C': [0, 1, 0, 0],
                          'G': [0, 0, 1, 0],
                          'T': [0, 0, 0, 1],
                          '-': [1, 1, 1, 1]}
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

    if hohna_data:
        datadict_raw = pd.read_pickle('data/hohna_datasets/DS1.pickle')
        genome_strings = list(datadict_raw.values())
        datadict = form_dataset_from_strings(genome_strings, Alphabet_dir_blank)
        # print(datadict['genome'].shape)


    if corona_data:
        #max_site = 300
        datadict_raw = pd.read_pickle('data/betacoronavirus_p4.pickle')
        datadict = {}
        datadict['taxa'] = datadict_raw['taxa']
        datadict['genome'] = np.array(datadict_raw['genome'])[:, 10000:10100, :]
        # print(datadict['taxa'])
        # for i in range(100):
        #     print(datadict['genome'][0,i,:])

    if primate_data:
        datadict_raw = pd.read_pickle('data/primate.p')
        genome_strings = list(datadict_raw.values())
        datadict = form_dataset_from_strings(genome_strings, Alphabet_dir_blank)


    if simulate_data:
        data_NxSxA = simulateDNA(3, 5, alphabet)
        # print("Simulated genomes:\n", data_NxSxA)
        taxa = ['S' + str(i) for i in range(data_NxSxA.shape[0])]
        datadict = {'taxa': taxa,
                    'genome': data_NxSxA}


    if load_strings:
        genome_strings = ['ACTTTGAGAG', 'ACTTTGACAG', 'ACTTTGACTG', 'ACTTTGACTC']
        datadict = form_dataset_from_strings(genome_strings, Alphabet_dir)


    #pdb.set_trace()
    vcsmc = vcsmc.VCSMC(datadict,K=128)

    vcsmc.train(epochs=args.num_epoch, batch_size=args.batch_size, learning_rate=args.learning_rate, memory_optimization=args.memory_optimization)
