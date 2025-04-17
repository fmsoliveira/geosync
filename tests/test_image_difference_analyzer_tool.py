import json
import logging

from geosync.tools.image_difference_analyzer_tool import ImageDifferenceAnalyzerTool

if __name__ == "__main__":
    print("--- Iniciando teste isolado da Tool ---")
    
    tool = ImageDifferenceAnalyzerTool()

    input_json = json.dumps({
        "image_path_1": "raw_images/satellite_image_38.5732763_-7.9058921_2025-04-08.zip",
        "image_path_2": "raw_images/satellite_image_38.5732763_-7.9058921_2025-04-15.zip"
    })

    try:
        print("A executar a tool com input:")
        logging.info("A executar a tool com input:")

        params = json.loads(input_json)
        resultado = tool._run(**params)

        print("--- Resultado da Tool ---")
        print(resultado)
        logging.info(f"Resultado da tool: {resultado}")
        
    except Exception as e:
        print(f"Erro durante a execução da tool: {e}")
        logging.error("Erro durante a execução da tool:", exc_info=True)

    print("--- Fim do teste isolado da Tool ---")