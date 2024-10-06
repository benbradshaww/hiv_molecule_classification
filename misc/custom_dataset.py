from torch_geometric.data import InMemoryDataset


class CustomDataset(InMemoryDataset):
    def __init__(self, data_list, transform=None):
        super().__init__(None, transform)
        self.data, self.slices = self.collate(data_list)

    @property
    def processed_file_names(self):
        return ["dummy.pt"]

    def process(self):
        pass
