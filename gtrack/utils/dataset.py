import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import IterableDataset
import numpy as np
from toytrack import ParticleGun, Detector, EventGenerator
from typing import Optional, Union, List

class TracksDataset(IterableDataset):
    def __init__(
            self,
            hole_inefficiency: Optional[float] = 0,
            d0: Optional[float] = 0.1,
            noise: Optional[Union[float, List[float], List[Union[float, str]]]] = 0,
            minbias_num_dist: Optional[Union[float, List[float], List[Union[float, str]]]] = [50, None, 'poisson'],
            minbias_pt_dist: Optional[Union[float, List[float], List[Union[float, str]]]] = [1, 10],
            pileup_num_dist: Optional[Union[float, List[float], List[Union[float, str]]]] = [50, None, 'poisson'],
            pileup_pt_dist: Optional[Union[float, List[float], List[Union[float, str]]]] = [1, 10],
            hard_proc_num_dist: Optional[Union[float, List[float], List[Union[float, str]]]] = [5, None, 'poisson'],
            hard_proc_pt_dist: Optional[Union[float, List[float], List[Union[float, str]]]] = [100, 5, 'normal'],
        ):
        super().__init__()

        self.noise = noise

        detector = Detector(
            dimension=2,
            hole_inefficiency=hole_inefficiency
        ).add_from_template(
            'barrel', 
            min_radius=0.5, 
            max_radius=3, 
            number_of_layers=10,
        )
        
        minbias = ParticleGun(
            dimension=2, 
            num_particles=minbias_num_dist, 
            pt=minbias_pt_dist, 
            pphi=[-np.pi, np.pi], 
            vx=[0, d0 * 0.5**0.5, 'normal'], 
            vy=[0, d0 * 0.5**0.5, 'normal'],
        )

        pileup = ParticleGun(
            dimension=2, 
            num_particles=pileup_num_dist, 
            pt=pileup_pt_dist, 
            pphi=[-np.pi, np.pi], 
            vx=[0, d0 * 0.5**0.5, 'normal'], 
            vy=[0, d0 * 0.5**0.5, 'normal'],
        )

        hard_proc = ParticleGun(
            dimension=2, 
            num_particles=hard_proc_num_dist, 
            pt=hard_proc_pt_dist, 
            pphi=[-np.pi, np.pi], 
            vx=[0, d0 * 0.5**0.5, 'normal'], 
            vy=[0, d0 * 0.5**0.5, 'normal'],
        )

        self.minbias_gen = EventGenerator(minbias, detector, noise)
        self.hard_proc_gen = EventGenerator([pileup, hard_proc], detector, noise)

    def __iter__(self):
        return self
    
    def __next__(self):
        y = (np.random.rand() < 0.5)
        if y:
            event = self.hard_proc_gen.generate_event()
        else:
            event = self.minbias_gen.generate_event()
        x = torch.tensor([event.hits.x, event.hits.y], dtype=torch.float).T.contiguous()
        mask = torch.ones(x.shape[0], dtype=bool)
        return x, mask, torch.tensor([y], dtype=torch.float), event
    
def collate_fn(ls):
    x, mask, y, events = zip(*ls)
    return pad_sequence(x, batch_first=True), pad_sequence(mask, batch_first=True), torch.cat(y), events
