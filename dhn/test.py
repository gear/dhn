from datasets import HomDataset, HomDataLoader
from models import DHN


def test_enzymes(path):
    enzymzes = HomDataset('ENZYMES', root_path=path)
    assert len(enzymzes) == 600
    assert enzymzes.num_classes == 6, print(enzymzes.num_classes)
    assert enzymzes.num_features == 21    

    loader = HomDataLoader(enzymzes, batch_size=10, shuffle=True)
    for batch in loader:
        print(batch.x.shape)
        print(batch.y.shape)
        print(batch.mapping_index_dict.keys())

    for batch in loader:
        batch = batch.to('cuda')
        print(batch.x.device)
        print(batch.y.device)
        print([t.device for t in batch.mapping_index_dict.values()])
        

def test_dhn(**kwargs):
    model = DHN(**kwargs)
    print(model)

if __name__ == '__main__':
    data_path = '/'.join(__file__.split('/')[:-3])+'/data'
    test_enzymes(path=data_path)
    test_dhn(agg=[2, 4])