# tests/test_urban_analysis_tool.py
from geosync.tools.urban_analysis_tool import UrbanGrowthAnalyzerTool

def test_urban_growth_analysis():
    """
    Testa a ferramenta de análise de crescimento urbano usando duas imagens de satélite
    """
    # Criar instância da ferramenta
    analyzer = UrbanGrowthAnalyzerTool()
    
    # Caminhos para imagens de teste
    # Substitua pelos caminhos corretos das suas imagens de teste
    image_path_1 = "output/rgb_antiga.png"
    image_path_2 = "output/rgb_recente.png"
    
    # Executar a análise
    try:
        results = analyzer._run(image_path_1=image_path_1, image_path_2=image_path_2)
        
        # Imprimir resultados
        print("\n===== Resultados da Análise =====")
        print(f"Edifícios na imagem 1: {results['Edifícios na data 1']}")
        print(f"Edifícios na imagem 2: {results['Edifícios na data 2']}")
        print(f"Novas construções: {results['Novas construções']}")
        print(f"Imagem de diferença salva em: {results['Imagem de diferença guardada em']}")
        print("================================\n")
        
        return True
    except Exception as e:
        print(f"Erro durante o teste: {str(e)}")
        return False

if __name__ == "__main__":
    test_urban_growth_analysis()