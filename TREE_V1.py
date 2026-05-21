
import rasterio
import cv2
import numpy as np
import geopandas as gpd
from shapely.geometry import Point, Polygon
import os
import sys

# =============================================================================
# CONFIGURAÇÕES GLOBAIS (AJUSTE CONFORME SEU AMBIENTE)
# =============================================================================

# Caminho para a imagem GeoTIFF de entrada (mosaico de drone)
INPUT_IMAGE_PATH = r"D:\TESTES_PYTHON\OPEM_CV_TESTE\INPUT\teste_pilha.tif"

# Caminho para o arquivo GeoJSON/Shapefile de pontos de treino (Cenário 2)
# Deixe como None ou vazio se for usar apenas o Cenário 1
TRAINING_POINTS_PATH = r"D:\TESTES_PYTHON\OPEM_CV_TESTE\INPUT\treino.geojson"
TRAINING_POINTS_PATH = ""

# Caminho para o diretório de saída dos resultados
OUTPUT_DIR = r"D:\TESTES_PYTHON\OPEM_CV_TESTE\resultados"

# Nome do arquivo GeoJSON de saída (polígonos das árvores)
OUTPUT_POLYGONS_FILENAME = "arvores_detectadas.geojson"

# =============================================================================
# FUNÇÕES DE PROCESSAMENTO
# =============================================================================

def detect_tree_polygons(
    tiff_path,
    output_geojson_path,
    training_points_path=None,
    default_hsv_lower=np.array([35, 50, 40]),
    default_hsv_upper=np.array([85, 255, 255]),
    min_area=100,
    max_area=8000,
    min_circularity=0.4
):
    """
    Detecta polígonos de árvores em uma imagem GeoTIFF e os salva como GeoJSON.
    Pode usar pontos de treino para refinar os parâmetros de cor.
    """

    if not os.path.exists(tiff_path):
        print(f"Erro: Imagem de entrada não encontrada em {tiff_path}")
        return
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    with rasterio.open(tiff_path) as src:
        transform = src.transform
        crs = src.crs
        print(f"Processando imagem: {tiff_path}")
        print(f"CRS detectado: {crs}")

        # Ler bandas RGB (1, 2, 3). Se não tiver 3 bandas, tenta ler a primeira e replica.
        try:
            img_data = src.read([1, 2, 3])
        except Exception:
            print("Aviso: Imagem não possui 3 bandas RGB. Tentando ler como escala de cinza.")
            img_data = src.read(1)
            img_data = np.stack([img_data, img_data, img_data]) # Replicar para 3 canais

        img_rgb = np.moveaxis(img_data, 0, -1)
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

    # 1. Pré-processamento: Filtro Bilateral para reduzir ruído preservando bordas
    blur = cv2.bilateralFilter(img_bgr, 9, 75, 75)
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

    # 2. Determinar parâmetros HSV (Cenário 1 ou 2)
    current_hsv_lower = default_hsv_lower
    current_hsv_upper = default_hsv_upper

    if training_points_path and os.path.exists(training_points_path):
        print(f"Cenário 2: Refinando parâmetros HSV com pontos de treino de {training_points_path}")
        try:
            training_gdf = gpd.read_file(training_points_path)
            if training_gdf.empty:
                print("Aviso: Arquivo de pontos de treino vazio. Usando parâmetros HSV padrão.")
            else:
                h_values, s_values, v_values = [], [], []
                for idx, row in training_gdf.iterrows():
                    if row.geometry.geom_type == 'Point':
                        # Converter coordenada geográfica do ponto de treino para pixel
                        px_x, px_y = src.index(row.geometry.x, row.geometry.y)
                        
                        # Amostrar HSV em uma pequena janela ao redor do ponto
                        # Garantir que a janela esteja dentro dos limites da imagem
                        half_window = 5 # Tamanho da janela 11x11 pixels
                        y_min = max(0, int(px_y) - half_window)
                        y_max = min(hsv.shape[0], int(px_y) + half_window + 1)
                        x_min = max(0, int(px_x) - half_window)
                        x_max = min(hsv.shape[1], int(px_x) + half_window + 1)

                        if y_max > y_min and x_max > x_min:
                            patch = hsv[y_min:y_max, x_min:x_max]
                            if patch.size > 0:
                                h_values.extend(patch[:,:,0].flatten())
                                s_values.extend(patch[:,:,1].flatten())
                                v_values.extend(patch[:,:,2].flatten())
                
                if h_values and s_values and v_values:
                    # Calcular média e desvio padrão para definir o range HSV
                    h_mean, h_std = np.mean(h_values), np.std(h_values)
                    s_mean, s_std = np.mean(s_values), np.std(s_values)
                    v_mean, v_std = np.mean(v_values), np.std(v_values)

                    # Definir range HSV dinamicamente (ajustar os multiplicadores conforme necessário)
                    current_hsv_lower = np.array([
                        max(0, int(h_mean - 2 * h_std)),
                        max(0, int(s_mean - 2 * s_std)),
                        max(0, int(v_mean - 2 * v_std))
                    ])
                    current_hsv_upper = np.array([
                        min(179, int(h_mean + 2 * h_std)), # H vai de 0-179 no OpenCV
                        min(255, int(s_mean + 2 * s_std)),
                        min(255, int(v_mean + 2 * v_std))
                    ])
                    print(f"HSV range refinado: Lower={current_hsv_lower}, Upper={current_hsv_upper}")
                else:
                    print("Aviso: Não foi possível amostrar HSV dos pontos de treino. Usando parâmetros padrão.")
        except Exception as e:
            print(f"Erro ao ler pontos de treino ou refinar HSV: {e}. Usando parâmetros HSV padrão.")
    else:
        print("Cenário 1: Usando parâmetros HSV padrão.")

    # 3. Segmentação de cor com o range HSV determinado
    mask = cv2.inRange(hsv, current_hsv_lower, current_hsv_upper)

    # 4. Operações morfológicas para limpar e separar objetos
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel) # Remove pequenos ruídos
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel) # Fecha pequenos buracos

    # 5. Encontrar contornos
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    polygons = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)

        if perimeter == 0: continue

        circularity = 4 * np.pi * area / (perimeter * perimeter)

        # 6. Filtrar contornos por área e circularidade
        if min_area < area < max_area and circularity > min_circularity:
            # Converter contorno de pixel para coordenadas geográficas
            # O contorno é uma lista de pontos (x, y) de pixel
            # Precisamos converter cada ponto do contorno para geo-coordenadas
            geo_coords = []
            for point in cnt.reshape(-1, 2):
                lon, lat = rasterio.transform.xy(transform, point[1], point[0])
                geo_coords.append((lon, lat))
            
            # Criar polígono Shapely a partir dos pontos geográficos
            if len(geo_coords) >= 3: # Polígono precisa de no mínimo 3 pontos
                poly = Polygon(geo_coords)
                if not poly.is_valid:
                    poly = poly.convex_hull
                if poly.is_valid and poly.area > 0:
                    polygons.append(poly)

    if polygons:
        # Criar GeoDataFrame e salvar como GeoJSON
        gdf = gpd.GeoDataFrame(geometry=polygons, crs=crs)
        final_output_path = output_geojson_path
        gdf.to_file(final_output_path, driver='GeoJSON')
        print(f"Detecção concluída: {len(polygons)} polígonos de árvores salvos em {final_output_path}")
    else:
        print("Nenhum polígono de árvore detectado com os parâmetros atuais.")

# =============================================================================
# EXECUÇÃO PRINCIPAL
# =============================================================================

if __name__ == "__main__":
    # Exemplo de uso para Cenário 1 (apenas busca)
    print("\n--- Executando Cenário 1 (Busca Cega) ---")
    detect_tree_polygons(
        tiff_path=INPUT_IMAGE_PATH,
        output_geojson_path=os.path.join(OUTPUT_DIR, "arvores_cenario1.geojson"),
        training_points_path=None # Nulo para Cenário 1
    )

    # Exemplo de uso para Cenário 2 (com pontos de treino)
    if TRAINING_POINTS_PATH and os.path.exists(TRAINING_POINTS_PATH):
        print("\n--- Executando Cenário 2 (Refinamento com Pontos de Treino) ---")
        detect_tree_polygons(
            tiff_path=INPUT_IMAGE_PATH,
            output_geojson_path=os.path.join(OUTPUT_DIR, "arvores_cenario2_refinado.geojson"),
            training_points_path=TRAINING_POINTS_PATH
        )
    else:
        print("\n--- Cenário 2 (Refinamento com Pontos de Treino) ignorado: Arquivo de treino não especificado ou não encontrado. ---")


