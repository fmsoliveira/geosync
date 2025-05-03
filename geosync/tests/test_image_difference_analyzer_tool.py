import json
import logging

from geosync.tools.image_difference_analyzer_tool import ImageDifferenceAnalyzerTool

if __name__ == "__main__":
    print("--- Iniciando teste isolado da Tool ---")
    
    tool = ImageDifferenceAnalyzerTool()

    input_json = json.dumps({
        "first_date_images": {
            "NE": "raw_images/satellite_image_44.1026_9.8241_2023-07-10_NE.zip",
            "NO": "raw_images/satellite_image_44.1026_9.8241_2023-07-10_NO.zip",
            "SO": "raw_images/satellite_image_44.1026_9.8241_2023-07-10_SE.zip",
            "SE": "raw_images/satellite_image_44.1026_9.8241_2023-07-10_SO.zip"
        },
        "second_date_images": {
            "NE": "raw_images/satellite_image_44.1026_9.8241_2025-04-30_NE.zip",
            "NO": "raw_images/satellite_image_44.1026_9.8241_2025-04-30_NO.zip",
            "SO": "raw_images/satellite_image_44.1026_9.8241_2025-04-30_SE.zip",
            "SE": "raw_images/satellite_image_44.1026_9.8241_2025-04-30_SO.zip"
        }
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