"""
Classe genérica para visualização de dados de tabelas OCR baseada no dataset ICDAR2019 cTDaR.

Este módulo fornece ferramentas para visualizar anotações de tabelas em imagens de documentos,
incluindo células, bordas e máscaras de região de tabela.

Referência: https://cndplab-founder.github.io/cTDaR2019/dataset-description.html
"""

import xml.etree.ElementTree as ET
from typing import List, Tuple, Dict, Optional, Union
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import PatchCollection
from PIL import Image, ImageDraw
import cv2


@dataclass
class Cell:
    """Representa uma célula de tabela com suas coordenadas e metadados."""
    cell_id: str
    coordinates: List[Tuple[int, int]]
    start_row: Optional[int] = None
    end_row: Optional[int] = None
    start_col: Optional[int] = None
    end_col: Optional[int] = None
    
    def get_bbox(self) -> Tuple[int, int, int, int]:
        """Retorna a bounding box (x_min, y_min, x_max, y_max) da célula."""
        x_coords = [coord[0] for coord in self.coordinates]
        y_coords = [coord[1] for coord in self.coordinates]
        return (min(x_coords), min(y_coords), max(x_coords), max(y_coords))
    
    def get_polygon(self) -> np.ndarray:
        """Retorna os pontos do polígono como array numpy."""
        return np.array(self.coordinates, dtype=np.int32)


@dataclass
class Table:
    """Representa uma tabela com suas células e metadados."""
    table_id: str
    coordinates: List[Tuple[int, int]]
    cells: List[Cell]
    
    def get_bbox(self) -> Tuple[int, int, int, int]:
        """Retorna a bounding box (x_min, y_min, x_max, y_max) da tabela."""
        x_coords = [coord[0] for coord in self.coordinates]
        y_coords = [coord[1] for coord in self.coordinates]
        return (min(x_coords), min(y_coords), max(x_coords), max(y_coords))
    
    def get_polygon(self) -> np.ndarray:
        """Retorna os pontos do polígono como array numpy."""
        return np.array(self.coordinates, dtype=np.int32)


class TableAnnotationParser:
    """Parser para arquivos XML de anotação de tabelas no formato ICDAR2019 cTDaR."""
    
    @staticmethod
    def parse_coordinates(coord_string: str) -> List[Tuple[int, int]]:
        """
        Parse uma string de coordenadas no formato "x1,y1 x2,y2 x3,y3 ..."
        
        Args:
            coord_string: String contendo as coordenadas
            
        Returns:
            Lista de tuplas (x, y)
        """
        coords = []
        points = coord_string.strip().split()
        for point in points:
            x, y = map(int, point.split(','))
            coords.append((x, y))
        return coords
    
    @staticmethod
    def parse_xml(xml_path: Union[str, Path]) -> Tuple[str, List[Table]]:
        """
        Parse um arquivo XML de anotação de tabela.
        
        Args:
            xml_path: Caminho para o arquivo XML
            
        Returns:
            Tupla contendo (nome_do_arquivo, lista_de_tabelas)
        """
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        # Obter o nome do arquivo da imagem
        filename = root.get('filename', 'unknown')
        
        tables = []
        for table_elem in root.findall('table'):
            table_id = table_elem.get('id', 'unknown')
            
            # Parse das coordenadas da tabela
            table_coords_elem = table_elem.find('Coords')
            if table_coords_elem is not None:
                table_coords = TableAnnotationParser.parse_coordinates(
                    table_coords_elem.get('points', '')
                )
            else:
                table_coords = []
            
            # Parse das células
            cells = []
            for cell_elem in table_elem.findall('cell'):
                cell_id = cell_elem.get('id', 'unknown')
                start_row = cell_elem.get('start-row') 
                end_row = cell_elem.get('end-row')
                start_col = cell_elem.get('start-col')
                end_col = cell_elem.get('end-col')
                
                # Converter para int se existir
                start_row = int(start_row) if start_row is not None else None
                end_row = int(end_row) if end_row is not None else None
                start_col = int(start_col) if start_col is not None else None
                end_col = int(end_col) if end_col is not None else None
                
                # Parse das coordenadas da célula
                cell_coords_elem = cell_elem.find('Coords')
                if cell_coords_elem is not None:
                    cell_coords = TableAnnotationParser.parse_coordinates(
                        cell_coords_elem.get('points', '')
                    )
                else:
                    cell_coords = []
                
                cell = Cell(
                    cell_id=cell_id,
                    coordinates=cell_coords,
                    start_row=start_row,
                    end_row=end_row,
                    start_col=start_col,
                    end_col=end_col
                )
                cells.append(cell)
            
            table = Table(
                table_id=table_id,
                coordinates=table_coords,
                cells=cells
            )
            tables.append(table)
        
        return filename, tables


class TableVisualizer:
    """Classe para visualização de anotações de tabelas em imagens."""
    
    def __init__(self, figsize: Tuple[int, int] = (15, 10)):
        """
        Inicializa o visualizador.
        
        Args:
            figsize: Tamanho da figura matplotlib (largura, altura)
        """
        self.figsize = figsize
    
    def visualize_tables(
        self,
        tables: List[Table],
        image: Optional[Union[str, Path, np.ndarray, Image.Image]] = None,
        show_cells: bool = True,
        show_table_region: bool = True,
        show_cell_grid: bool = True,
        table_color: str = 'red',
        cell_color: str = 'blue',
        alpha: float = 0.3,
        line_width: int = 2,
        save_path: Optional[Union[str, Path]] = None,
        title: Optional[str] = None
    ) -> plt.Figure:
        """
        Visualiza tabelas e células sobre uma imagem.
        
        Args:
            tables: Lista de objetos Table
            image: Imagem (caminho, array numpy ou PIL Image). Se None, cria imagem branca automaticamente.
            show_cells: Se True, mostra as células individuais
            show_table_region: Se True, mostra a região da tabela
            show_cell_grid: Se True, desenha grades das células
            table_color: Cor para a região da tabela
            cell_color: Cor para as células
            alpha: Transparência dos polígonos (0-1)
            line_width: Largura das linhas
            save_path: Caminho para salvar a figura (opcional)
            title: Título da figura (opcional)
            
        Returns:
            Figura matplotlib
        """
        # Carregar ou criar imagem
        if image is None:
            # Criar imagem branca baseada nas coordenadas das tabelas
            if tables:
                max_x = max_y = 0
                for table in tables:
                    if table.coordinates:
                        bbox = table.get_bbox()
                        max_x = max(max_x, bbox[2])
                        max_y = max(max_y, bbox[3])
                # Adicionar margem
                img_array = np.ones((max_y + 100, max_x + 100, 3), dtype=np.uint8) * 255
            else:
                # Se não há tabelas, criar imagem padrão
                img_array = np.ones((800, 600, 3), dtype=np.uint8) * 255
        elif isinstance(image, (str, Path)):
            img = Image.open(image)
            img_array = np.array(img)
        elif isinstance(image, Image.Image):
            img_array = np.array(image)
        else:
            img_array = image
        
        # Criar figura
        fig, ax = plt.subplots(figsize=self.figsize)
        ax.imshow(img_array)
        
        # Desenhar cada tabela
        for table in tables:
            # Desenhar região da tabela
            if show_table_region and table.coordinates:
                table_poly = patches.Polygon(
                    table.coordinates,
                    linewidth=line_width,
                    edgecolor=table_color,
                    facecolor=table_color,
                    alpha=alpha,
                    label=f'Table {table.table_id}'
                )
                ax.add_patch(table_poly)
            
            # Desenhar células
            if show_cells:
                for cell in table.cells:
                    if cell.coordinates:
                        if show_cell_grid:
                            # Desenhar apenas as bordas da célula
                            cell_poly = patches.Polygon(
                                cell.coordinates,
                                linewidth=1,
                                edgecolor=cell_color,
                                facecolor='none',
                                alpha=1.0
                            )
                        else:
                            # Desenhar célula preenchida
                            cell_poly = patches.Polygon(
                                cell.coordinates,
                                linewidth=1,
                                edgecolor=cell_color,
                                facecolor=cell_color,
                                alpha=alpha
                            )
                        ax.add_patch(cell_poly)
        
        # Configurar eixos e título
        ax.axis('off')
        if title:
            ax.set_title(title, fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        # Salvar se especificado
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    