import os
import re
from glob import glob
from pathlib import Path
from typing import Iterable
from tqdm import tqdm

class FileFinder: 

    @staticmethod
    def find_files(dir_path : str | Path, 
                   format_list : Iterable[str], 
                   sort : bool = True):
        
        files = []
        for format in format_list:
            files.extend(glob(os.path.join(dir_path, f"*.{format}")))
        
        if sort: 
            return sorted(files)
        
        return files
    
    @staticmethod
    def associate_files_by_name(first_files_list : Iterable[str | Path], 
                                second_files_list : Iterable[str | Path]):
        
        
        pairs = []


        for first_file in tqdm(first_files_list, desc='Associando pares... '):
            
            first_file_basename = os.path.splitext(os.path.basename(first_file))[0]

            matches = [second_file for second_file in second_files_list 
                       if first_file_basename == os.path.splitext(os.path.basename(second_file))[0]] # melhorar
            
            for match in matches: 
                if os.path.exists(match):
                    pairs.append((first_file, match))

        return pairs