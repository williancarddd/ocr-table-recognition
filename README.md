# ocr-table-recognition

Reporistório do projeto da disciplina Redes Neurais e Aprendizado profundo (SCC5809) do ICMC-USP pelos alunos:

- Lucas Ferro Zampar
- William Cardoso Barbosa
- Bruna Campos Guedes
- João Victor de Castro Oliveira

O foco do projeto é o reconhecimento da estrutura interna de tabelas empregando os modelos YOLOv8 e YOLOv11 para segmentação de instância no conjunto ICDAR-2019 e detecção de objetos no conjunto FinTabNet. O treinamento dos modelos foi realizado por meio do notebook `training.ipynb` no ambiente do Google ColaboratoryPro utilizando a GPU A100 com 80 Gb de VRAM. A geração dos respectivos conjuntos de dados foi realizada pelos notebooks intitulados `dataset_generation_*.ipynb`. Códigos de apoio foram desenvolvidos nos módulos DataExtractor, DataAugmentation e DataVisualization com foco principal no processo de ETL dos conjuntos originais. Uma vez formatados, os conjuntod foram carregados no GoogleDrive para importação na VM do Google Colab. 
