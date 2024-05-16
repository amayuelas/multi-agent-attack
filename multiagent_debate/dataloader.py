from glob import glob
import pandas as pd
import json
import random 

random.seed(0)

class MMLU:
    def __init__(self, dataset_name='mmlu', n_samples=50, data_dir='data'):
        self.dataset_name = dataset_name
        self.data_path = f'{data_dir}/{dataset_name}/test/*.csv'
        self.data_files = glob(self.data_path)
        self.data = None
        self.n_samples = n_samples
        self.selected_data = None
        self.load_data()

    def load_data(self):
        
        dfs = [pd.read_csv(file, header=None) for file in self.data_files]
        self.data = pd.concat(dfs, ignore_index=True)
        self.selected_data = self.data.sample(self.n_samples)
            

    def __getitem__(self, idx):
        return self.selected_data.iloc[idx].to_dict()

    def __len__(self):
        return len(self.selected_data)


class Math:
    def __init__(self, dataset_name='math', n_samples=50, data_dir='data', level_geq=1):
        self.dataset_name = dataset_name        
        self.data_path = f'{data_dir}/{dataset_name}/test'
        self.data_categories_path = glob(self.data_path + '/*')
        self.level_geq = level_geq
        self.data = None
        self.n_samples = n_samples
        self.selected_data = None
        self.load_data()
        self.filter_data()

    def load_data(self):

        self.data = []
        for category in self.data_categories_path:
            data_files = glob(category + '/*.json')
            for file in data_files:
                with open(file, 'r') as f:
                    self.data.append(json.load(f))
        self.data = pd.DataFrame(self.data)
        self.data['level'] = self.data['level'].apply(lambda x: int(x.split(' ')[1]))

    def filter_data(self):
        self.selected_data = self.data[self.data['level'] >= self.level_geq].sample(self.n_samples)
            
    def __getitem__(self, idx):
        return self.selected_data.iloc[idx].to_dict()

    def __len__(self):
        return len(self.selected_data)



class Chess:
    def __init__(self, dataset_name='chess', n_samples=50, data_dir='data'):
        self.dataset_name = dataset_name
        self.data_path = f'{data_dir}/{dataset_name}/synthetic_short/*.json'
        self.data_files = glob(self.data_path)
        self.data = None
        self.n_samples = n_samples
        self.selected_data = None
        self.load_data()

    def load_data(self):
        self.data = json.load(open(self.data_files[0], 'r'))
        self.data = pd.DataFrame(self.data['examples'])
        self.selected_data = self.data.sample(self.n_samples)    

    def __getitem__(self, idx):
        return self.selected_data.iloc[idx].to_dict()

    def __len__(self):
        return len(self.selected_data)
    


class MQuake: 
    def __init__(self, dataset_name='mquake', n_samples=50, data_dir='data'):
        self.dataset_name = dataset_name
        self.data_path = f'{data_dir}/{dataset_name}'
        self.data_file = self.data_path + '/MQuAKE-CF-3k.json'
        self.data = None
        self.n_samples = n_samples
        self.selected_data = None
        self.load_data()

    def load_data(self):
        self.data = json.load(open(self.data_file, 'r'))
        self.data = pd.DataFrame(self.data)
        self.selected_data = self.data.sample(self.n_samples)    

    def __getitem__(self, idx):
        return self.selected_data.iloc[idx].to_dict()

    def __len__(self):
        return len(self.selected_data)
    

class Musique:
    def __init__(self, dataset_name='musique', n_samples=50, data_dir='data'):
        self.dataset_name = dataset_name
        self.data_path = f'{data_dir}/{dataset_name}'
        self.data_file = self.data_path + '/musique_ans_v1.0_dev_new_question_decomposition.jsonl'
        self.data = None
        self.n_samples = n_samples
        self.selected_data = None
        self.load_data()
    
    def load_data(self):
        self.data = pd.read_json(self.data_file, lines=True)
        self.selected_data = self.data.sample(self.n_samples)
    
    def __getitem__(self, idx):
        return self.selected_data.iloc[idx].to_dict()
    
    def __len__(self):
        return len(self.selected_data)
        

class TruthfulQA:
    def __init__(self, dataset_name='truthfulqa', n_samples=50, data_dir='data'):
        self.dataset_name = dataset_name
        self.data_path = f'{data_dir}/{dataset_name}'
        self.data_file = self.data_path + '/mc_task.json'
        self.data = None
        self.n_samples = n_samples
        self.selected_data = None
        self.load_data()
    
    def load_data(self):
        self.data = pd.read_json(self.data_file)
        self.selected_data = self.data.sample(self.n_samples)
    
    def __getitem__(self, idx):
        return self.selected_data.iloc[idx].to_dict()
    
    def __len__(self):
        return len(self.selected_data)


def get_dataset(dataset_name='mmlu', n_samples=50, data_dir='data'):

    if dataset_name == 'mmlu':
        return MMLU(dataset_name, n_samples, data_dir)
    elif dataset_name == 'math':
        return Math(dataset_name, n_samples, data_dir)
    elif dataset_name == 'chess':
        return Chess(dataset_name, n_samples, data_dir)
    elif dataset_name == 'mquake':
        return MQuake(dataset_name, n_samples, data_dir)
    elif dataset_name == 'musique':
        return Musique(dataset_name, n_samples, data_dir)
    elif dataset_name == 'truthfulqa':
        return TruthfulQA(dataset_name, n_samples, data_dir)
    else:
        raise ValueError(f"Dataset {dataset_name} not supported")
        