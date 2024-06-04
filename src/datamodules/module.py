from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

class LFDataSet():
    def __init__(
        self,
        lf_name: str,
        path: str,
        read_mode: str,
        anchor_mode: str,
        angular_resolution: int
    ):
        self._lf_name = lf_name
        self._path = path
        self._read_mode = read_mode
        self._anchor_mode = anchor_mode
        self._angular_resolution = angular_resolution
        self._order = []
        for row in range(1,self._angular_resolution+1):
            for col in range(1,self._angular_resolution+1):
                self._order.append([row,col])
        

    def collate_val(self, index):
        row, col = index[0]
        return {
            'rand_coord': [row, col]
        }
        

    def dataloader_val(self):
        loader = DataLoader(
            self._order,
            batch_size= 1,
            collate_fn= self.collate_val,
            shuffle= False,
            num_workers=5
        )
        return loader
    

class CNetworkDataModule(LightningDataModule):
    def __init__(
        self,
        lf_name: str,
        path: str,
        read_mode: str,
        anchor_mode: str,
        angular_resolution: int
    ):
        
        super().__init__()
        self.dataset = LFDataSet(
            lf_name= lf_name,
            path= path,
            read_mode= read_mode,
            anchor_mode= anchor_mode,
            angular_resolution= angular_resolution
        )
        
    
    def train_dataloader(self) -> DataLoader:
        return self.dataset.dataloader()
    
    def val_dataloader(self) -> DataLoader:
        return self.dataset.dataloader_val()
    
    def test_dataloader(self) -> DataLoader:
        return self.dataset.dataloader_val()