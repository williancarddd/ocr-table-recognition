Para rodar a conversão de dados instale os requerimentos em requirements.txt com pip install -r requirements.txt. de preferencia em um ambiente virutal
Depois execute main_converter.py com o comando python main_converter.py, certifique de estar na pasta DataExtractor

O script vai:

Baixar automaticamente o dataset ICDAR2019 cTDaR (~5GB) e descompactar. essa etapa pode demorar um pouco
Converter para formato YOLO Segmentation em yolo/
Converter para formato Mask R-CNN (COCO) em rcnn/


Configurações:
Edite as configurações no main_converter.py se necessário:

train_ratio: proporção treino/validação (padrão: 0.8)
base_dir: diretório do dataset original
output_dir: diretório de saída


Sobre os conversores:
YOLOConverter: Gera anotações de segmentação no formato YOLO com coordenadas normalizadas
MaskRCNNConverter: Gera anotações no formato COCO JSON com segmentações poligonais

o script visu foi feito para testar se o codigo esta convertendo certo