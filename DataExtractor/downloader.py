import os
import zipfile
import urllib.request


def download_progress_hook(count, block_size, total_size):
    '''
    Função acionável que exibe o progresso de download dos dados. Caso total_size seja indefinido (-1), então
    exibirá apenas a quantidade de bytes baixadas até o momento. Count monitora a quantidade de blocos já baixados. 
    '''

    if total_size > 0:
        percent = min(100, (count * block_size * 100) / total_size)
        print(f"Baixado {count * block_size} de {total_size} bytes ({percent:.2f}%)", end='\r')
    else:
        print(f"Baixado {count * block_size} bytes de indefinido...", end='\r')

def download_dataset(reporthook=download_progress_hook):
    '''
    Realiza o download do dataset. 
    Inicialmente, o dataset é baixado no arquivo dataset.zip. 
    Em seguida, o conteúdo é extraído para o diretório ICDAR2019_cTDaR-master. 
    Por fim, o arquivo zip é excluído e o diretóiro renomeado para dataset.
    '''
    
    zip_dataset_url = "https://github.com/cndplab-founder/ICDAR2019_cTDaR/archive/refs/heads/master.zip"
    target_dataset_dir = "dataset"
    zip_dataset_file = "dataset.zip"
    temp_dir = "ICDAR2019_cTDaR-master"

    if os.path.exists(target_dataset_dir):
        print(f"{target_dataset_dir} já existe!")
        return
    
    print("Baixando o dataset... espere!")
    
    try:
        urllib.request.urlretrieve(zip_dataset_url, zip_dataset_file, reporthook=reporthook)
        print()
        print("Extraindo dataset...")
        
        with zipfile.ZipFile(zip_dataset_file, 'r') as zip_ref:
            zip_ref.extractall(".")
        
        os.rename(temp_dir, target_dataset_dir)
        os.remove(zip_dataset_file)
        print("Download do dataset!")
        
    except Exception as e:
        print(f"Erro: {e}")
        raise
