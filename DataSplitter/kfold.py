import pandas as pd
from sklearn.model_selection import KFold
from tqdm import tqdm

class DataFrameKFoldSplitter:
    def __init__(self, 
                 data : pd.DataFrame, 
                 n_splits : int = 5,
                 shuffle : bool = True, 
                 random_state : int = 42):

        # paris é o DataFrame com as colunas image_path e label_path 
        # a partir do qual a divisão em KFolds é realizada
        self.data = data  
        self.n_splits = n_splits
        self.shuffle = shuffle 
        self.random_state = random_state 
        

    def split_folds(self):
        '''
        Realiza a divisão do Dataset em KFolds. 
        '''
        
        # Define os folds do dataset. Nesse caso, é possível embaralhar os dados antes da divisão. 
        # Mesmo com o embaralhamento, não haverá repetições de dados entre as partições de validação. 
        kf = KFold(self.n_splits, shuffle=self.shuffle, random_state=self.random_state)
        kf.get_n_splits(self.data)

        folds = []
        
        # Para cada fold do dataset
        for train_index, val_index in tqdm(kf.split(self.data), total=self.n_splits, desc='Gerando folds...'):
            
            train_fold = self.data.iloc[train_index]
            val_fold = self.data.iloc[val_index]

            folds.append({'train': train_fold, 'val': val_fold})
          
        return folds
          
          