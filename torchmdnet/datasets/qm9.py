import torch
from torch_geometric.transforms import Compose
from torch_geometric.datasets import QM9 as QM9_geometric
from torch_geometric.nn.models.schnet import qm9_target_dict


class QM9(QM9_geometric):
    def __init__(self, root, transform=None, dataset_arg=None):
        assert dataset_arg is not None, (
            "Please pass the desired property to "
            'train on via "dataset_arg". Available '
            f'properties are {", ".join(qm9_target_dict.values())}.'
        )

        self.label = dataset_arg
        if dataset_arg == "mu":
            self.label = "dipole_moment"
        elif dataset_arg == "alpha":  # set this value as placeholder during pre-training
            self.label = "isotropic_polarizability"
        elif dataset_arg == "r2":
            self.label = "electronic_spatial_extent"
        elif dataset_arg in ["U0", "U"]:
            self.label = "energy_" + dataset_arg
        elif dataset_arg == "H":
            self.label = "enthalpy_H"
        elif dataset_arg == "G":
            self.label = "free_energy"
        elif dataset_arg == "cv":
            self.label = "heat_capacity"

        label2idx = dict(zip(qm9_target_dict.values(), qm9_target_dict.keys()))
        self.label_idx = label2idx[self.label]

        if transform is None:
            transform = self._filter_label
        else:
            transform = Compose([transform, self._filter_label])

        super(QM9, self).__init__(root, transform=transform)

    def get_atomref(self, max_z=100):
        atomref = self.atomref(self.label_idx)
        if atomref is None:
            return None
        if atomref.size(0) != max_z:
            tmp = torch.zeros(max_z).unsqueeze(1)
            idx = min(max_z, atomref.size(0))
            tmp[:idx] = atomref[:idx]
            return tmp
        return atomref

    def _filter_label(self, batch):
        batch.y = batch.y[:, self.label_idx].unsqueeze(1)
        return batch

    def download(self):
        super(QM9, self).download()

    def process(self):
        super(QM9, self).process()
