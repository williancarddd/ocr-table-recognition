"""
Guia Rápido de Uso - Visualizador de Tabelas OCR
================================================

Este guia mostra como usar a classe TableVisualizer para visualizar
anotações de tabelas do dataset ICDAR2019 cTDaR.
"""

from pathlib import Path
from table_visualizer import TableAnnotationParser, TableVisualizer
import matplotlib.pyplot as plt



# Criar pasta de saída
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)


# Parse e Visualização

def exemplo_basico():
    """Exemplo mais simples possível."""
    print("EXEMPLO 1: Uso Básico")
    print("-" * 50)
    
    # 1. Parse do arquivo XML
    xml_path = Path("ICDAR2019_cTDaR/samples/ground_truth/cTDaR_s104.xml")
    filename, tables = TableAnnotationParser.parse_xml(xml_path)
    
    print(f"Arquivo: {filename}")
    print(f"Tabelas: {len(tables)}")
    print(f"Células: {sum(len(t.cells) for t in tables)}")
    
    # 2. Tentar carregar a imagem real do XML
    image_path = "ICDAR2019_cTDaR/samples/ground_truth/cTDaR_s104.jpg"
    
    # 3. Visualizar (se não encontrar imagem, TableVisualizer cria uma automaticamente)
    visualizer = TableVisualizer()
    output_path = OUTPUT_DIR / "output_basico.png"
    fig = visualizer.visualize_tables(
        tables,
        image=image_path,
        generate_mask=True,
        save_path=str(output_path)
    )
    plt.close()
    print(f"✅ Salvo: {output_path}\n")






if __name__ == "__main__":
    print("\n" + "=" * 60)
    print(" VISUALIZADOR DE TABELAS OCR")
    print("=" * 60 + "\n")
    
    try:
        exemplo_basico()
        
        print("=" * 60)
        print("TODOS OS EXEMPLOS EXECUTADOS COM SUCESSO!")
        print("=" * 60)
        print(f"\nArquivos salvos em: {OUTPUT_DIR.absolute()}/")
        print("\nArquivos gerados:")
        for file in sorted(OUTPUT_DIR.glob("*")):
            size = file.stat().st_size / 1024  # KB
            print(f"  - {file.name} ({size:.1f} KB)")
        print()
        
    except Exception as e:
        print(f"\n Erro: {str(e)}")
        import traceback
        traceback.print_exc()
