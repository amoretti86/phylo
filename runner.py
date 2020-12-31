import vcsmc as vcsmc
import numpy as np
import argparse
import pandas as pd

# export KMP_DUPLICATE_LIB_OK=TRUE

if __name__ == "__main__":

    corona_data = False
    real_data_1 = False
    real_data_2 = False
    load_strings = False
    simulate_data = False
    simulate_load_data = False
    primate_data = False

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='load_strings')
    parser.add_argument('--memory_optimization', default='on')
    args = parser.parse_args()

    exec(args.dataset + ' = True')

    Alphabet_dir = {'A': [1, 0, 0, 0],
                    'C': [0, 1, 0, 0],
                    'G': [0, 0, 1, 0],
                    'T': [0, 0, 0, 1]}
    alphabet_dir = {'a': [1, 0, 0, 0],
                    'c': [0, 1, 0, 0],
                    'g': [0, 0, 1, 0],
                    't': [0, 0, 0, 1]}
    alphabet = np.array([[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 1.]])


    def simulateDNA(nsamples, seqlength, alphabet):
        genomes_NxSxA = np.zeros([nsamples, seqlength, alphabet.shape[0]])
        for n in range(nsamples):
            genomes_NxSxA[n] = np.array([random.choice(alphabet) for i in range(seqlength)])
        return genomes_NxSxA


    def form_dataset_from_strings(genome_strings, alphabet_dir):
        genomes_NxSxA = np.zeros([len(genome_strings), len(genome_strings[0]), len(alphabet_dir)])
        for i in range(genomes_NxSxA.shape[0]):
            for j in range(genomes_NxSxA.shape[1]):
                genomes_NxSxA[i, j] = alphabet_dir[genome_strings[i][j]]
        taxa = ['S' + str(i) for i in range(genomes_NxSxA.shape[0])]
        datadict = {'taxa': taxa,
                    'genome': genomes_NxSxA}
        return datadict


    if simulate_data:
        data_NxSxA = simulateDNA(3, 5, alphabet)
        # print("Simulated genomes:\n", data_NxSxA)
        taxa = ['S' + str(i) for i in range(data_NxSxA.shape[0])]
        datadict = {'taxa': taxa,
                    'genome': data_NxSxA}

    if simulate_load_data:
        df = pd.read_csv('rdm_3.csv')
        genome_strings = []
        for i in range(1,4):
            genome_strings.append(df.iat[0,1])
        datadict = form_dataset_from_strings(genome_strings, alphabet_dir)


    if load_strings:
        genome_strings = ['ACTTTGAGAG', 'ACTTTGACAG', 'ACTTTGACTG', 'ACTTTGACTC']
        datadict = form_dataset_from_strings(genome_strings, Alphabet_dir)

    if corona_data:
        #max_site = 300
        datadict_raw = pd.read_pickle('data/betacoronavirus4.pickle')
        datadict = {}
        datadict['taxa'] = datadict_raw['taxa']
        datadict['genome'] = np.array(datadict_raw['genome'])[:, 10000:10100, :]
        # print(datadict['taxa'])
        # for i in range(100):
        #     print(datadict['genome'][0,i,:])

    if primate_data:
        Alphabet_dir_blank = {'A': [1, 0, 0, 0, 0],
                              'C': [0, 1, 0, 0, 0],
                              'G': [0, 0, 1, 0, 0],
                              'T': [0, 0, 0, 1, 0],
                              '-': [0, 0, 0, 0, 1]}

        datadict_raw = pd.read_pickle('data/primate.p')
        # max_site = 898
        # datadict = {}
        # datadict['taxa'] = datadict_raw.keys()
        # datadict['genome'] = np.array(datadict_raw.values())
        genome_strings = list(datadict_raw.values())
        datadict = form_dataset_from_strings(genome_strings, Alphabet_dir_blank)

    if real_data_1:
        genome_strings = \
            ['aaccctgttatttccacatgccaacaatcccaacag',
             'aactctgttatttccacatgccaacaatcccaacag',
             'aaatctgtgttgtctaaatgtcagttatttcagtta',
             'aaagctattatttaaaaatataaattatctcaatta',
             'aacactgttatttctaaatatcacttttcccaattg']
        datadict = form_dataset_from_strings(genome_strings, alphabet_dir)
        datadict['taxa'] = ['human', 'gibbon', 'guinea pig', 'aardvark', 'armadillo']

    if real_data_2:
        # Primates
        # block 19, 20, 32, 38, 42, 45, 47, 52, 53, 54, 57, 74, 78, 89, 92, 202, 223, 228, 239, 286, 304, 309, 346
        genome_strings = \
            [
                'taatggaataacacctttgctatgttatccaaacaatattagtcctttttcttctcttgtcgcccagccagagggcaatggtgggatctcggctcactgagacctctgcctcccagttcaagttacaggcacccgccaggctggtctcgaactgctgacctcaggtgatccacccaccttggcctccgaaagtgccgggattataggcgtgagccaccgcaccacctagcttgtatcgaacaaagggaataaaaaatgtatggatcaaggctcatgtacacaagatccaaattatccaccatccaggataatattttttgg',
                'gaatggaataacacctttgctatgttatccaaacaatattagtcctttttcttctcttgtcgcccagccagagggcaatggtgggatctccgctcactgagacctctgcctcccagttcaagttacaggcacccgccaggctggtctcgaactgctgacctcaggtgaaccacccaccttggcctccgaaagtgccgggattataggcgtgagccaccgcaccacctagcttgtatcgaacaaagggaataaaaaatgtatggatcaaggctcatgtacacaagatccaaattatccaccatccaggataatattttttgg',
                'taatggaataacacctttgctatgttatccaaacaatattagtccttttttttctcttgtcacccagccagagggcaatggcgggatctcggctcactgagacctctgcctcccagttcaagctacaggcacccgccaggctggtcttgaactgctgacctcaggtgatccacccaccttggcctccaaaagtgccgggattataggtgtgagccaccgcaccacctagcttgtatcgaactaagggaataaaaaatgtatggatcaaggctcatgtacacaagatccaaattatccaccatccaggataatatttttcgg',
                'taatggaataacacctttgctatgttatccaaacaatattagtcctattttttctcttgtcacccagccagagggcaatggtgggatctcggctcactgcgacctctgcctcccagttcaagctacaggcacccgccaggctgggctccaactgctgacctcaggtgatccacccatcttggcctccgaaagtgccgggattacaggcgtgagccaccgcactgcctagtttgtatcgaacaaagggaatataaaatgtatgattcaaggctcatgtacacaagatccaaattatcccccatccaggatagtattttacgg',
                'taatggaataacacctttgctatgttattcaaacaatattagtcctattttttctcttgttgcccagctggagggcaatggcgggatctcggctcgctgccacctctgcctcccagttcaggctacaggcacctgccatgctgttcctgaactgctgacctcaggtgatccacctaccttggcctccaaaagtgccgggattacaggcgtgagccaccgcactgcctagtttgtattgaacaaagggaatataaaatgtatgaatcaaggctcatgtacacaagatccaaattatccaccatccaggataatattttatgg',
                'taacagaataacacctttactatgttatctaaataatatttgtcctattttttctcttgtcacccagctggaaagcaatggcgggacctcagctcactgcaacctctgcctcccagttcaagctataggcatctgccaggctggtctcgaactgctgacctcaggtgatccacccgccttggcctcccaaagcgctgggattgtaggcatgagccaccccgccacctagtttgtatagaatataggagatacaaaatgtatgaatcaaggctgacgtatacacgatccaaattatcccccacccaggacaatattttctga',
                'taacagaataacacctttgctatgttatctaaataatatttgtcctattttttctcttgtcacccagctggaaagcaatggcgggacctcagctcactgcaacctctgcctcccagttcaagctacaggcatctgccaggctggtctcgaactgctgacctcaggtgatccacccgccttggcctcccaaagcgctgggattgtaggcatgagccaccccgccacctagtttgtatagaatataggagatacaaaatgtatgaatcaaggctgacgtatacacgatccaaattatcccccacccaggacaatattttctga',
                'taacagaataacacttttgctatgttatctaaataatatttgtcctatttcttctcttgtcgcccagctggaaggcaatggcgggacctcagctcactgcaacctctgcctcccagttcaagctacaggcatctgccaggctggtctagaactgctgacctcaggtgatccacccgccttggcctcccaaagtgctggaattgcaggcatgagccaccccgccacctagtttgtatagaatataggagatacaaaatgtatgaatcaaggctgacgtccacacgatccaaattatcccccacccaggacaatattttctga',
                'taacagaataacacctttgctatgttatctaaataatatttgtcctattttttctcttgtcgcccagctggtgggcaatggcggaatctcggctcaatgcaacctctgcctcccagttcaagctacaggcatctgtcaggctggtctcaaactgctgacctcaggtgatccacccgccttggcctcccaaagtgctgggattacaggcacgacccaccccgccacctagtttgtatagaatagaggagatacaaaatgtatgaatccaggctgacgtacacacgatccaaattatcccccacccaggacaatattttctga']
        datadict = form_dataset_from_strings(genome_strings, alphabet_dir)
        datadict['taxa'] = ['human', 'chimp', 'gorilla', 'oranguta', 'gibbon', 'rhesus', 'macaque', 'baboon',
                            'greenmonkey']


    #pdb.set_trace()
    vcsmc = vcsmc.VCSMC(datadict,K=16)

    vcsmc.train(epochs=100, batch_size=128, memory_optimization=args.memory_optimization)
