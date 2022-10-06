from tqdm.auto import trange

import numpy as np

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
import inspect

from copy import deepcopy


class TPPWrapper(Dataset):
    """
    Wrap data of a temporal point process
    """

    def __init__(self, lamb_func, n_sample, t_end, max_lamb,
                 fn=None, n_start=0, seed=0, verbose=False):
        self.lamb_func = lamb_func
        self.n_sample = n_sample
        self.t_end = t_end
        self.max_lamb = max_lamb
        self.seqs = []
        np.random.seed(seed)

        if fn is not None:
            self.seqs = [seq[seq < t_end] for seq in torch.load(fn)[n_start:n_start + n_sample]]
        else:
            for _ in trange(n_sample):
                self.seqs.append(torch.tensor(self.generate(verbose)))

    def save(self, name):
        torch.save(self.seqs, f'{name}.db')
        with open(f'{name}.info', 'w') as file:
            file.write(inspect.getsource(self.lamb_func) + '\n')
            file.write(f'n_sample = {self.n_sample}\n')
            file.write(f't_end = {self.t_end}\n')
            file.write(f'max_lamb = {self.max_lamb}\n')
            file.close()

    def generate(self, verbose=False):
        """
        Generate event timing sequence governed by temporal point process
        """
        if verbose:
            print(f'Generating events from t=0 to t={self.t_end}')

        t = 0.0
        his_t = np.array([])

        while True:
            # Calculate the maximum intensity
            lamb_t, L, M = self.lamb_func(t, his_t)
            delta_t = np.random.exponential(scale=1 / M)
            if lamb_t > self.max_lamb:  # Discarding the sequence
                return self.generate(verbose)
            if t + delta_t > self.t_end:
                break
            if delta_t > L:
                t += L
                continue
            else:
                t += delta_t
                new_lamb_t, _, _ = self.lamb_func(t, his_t)

                if new_lamb_t / M >= np.random.uniform():  # Accept the sample
                    if verbose:
                        print("----")
                        print(f"t:  {t}")
                        print(f"λt: {new_lamb_t}")
                    # Draw a location
                    his_t = np.append(his_t, t)

        return his_t

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        return self.seqs[idx]


def pad_collate(batch):
    seq_lens = [len(seq) for seq in batch]
    t_last = torch.tensor([seq[-1] for seq in batch])
    seq_pads = pad_sequence(batch, batch_first=True, padding_value=-1)
    return seq_pads.unsqueeze(-1), seq_lens, t_last
