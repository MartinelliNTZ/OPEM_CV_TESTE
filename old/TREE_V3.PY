
import rasterio
import cv2
import numpy as np
import geopandas as gpd
from shapely.geometry import Polygon, Point
import os
import sys

try:
    import pyogrio
except ImportError:
    pyogrio = None

# =============================================================================
# CONFIGURAÇÕES GLOBAIS (AJUSTE CONFORME SEU AMBIENTE)
# =============================================================================

# Caminho para a imagem GeoTIFF de entrada (mosaico de drone)
# Exemplo: r"C:\Users\SeuUsuario\Documentos\mosaico.tif"
INPUT_IMAGE_PATH = r"D:\TESTES_PYTHON\OPEM_CV_TESTE\imaru\Imaru2.tif"

# Caminho para o arquivo GeoJSON/Shapefile de pontos de treino (Cenário 2)
# Deixe como None ou vazio se for usar apenas o Cenário 1
TRAINING_POINTS_PATH = r"D:\TESTES_PYTHON\OPEM_CV_TESTE\imaru\floresta_pts.shp"
#TRAINING_POINTS_PATH = ""

# Caminho para o diretório de saída dos resultados
OUTPUT_DIR = r"D:\TESTES_PYTHON\OPEM_CV_TESTE\resultados3"

# Nomes de arquivos de treino comuns a procurar no diretório INPUT
DEFAULT_TRAINING_FILENAMES = [
    "PONTOS.geojson",
    "pontos.geojson",
    "treino.geojson",
    "floresta_pts.shp"
]

# Nome do arquivo GeoJSON de saída para o Cenário 1 (polígonos das árvores)
OUTPUT_POLYGONS_FILENAME_SCENARIO1 = "arvores_cenario1_exg_poligonos.geojson"

# Nome do arquivo GeoJSON de saída para o Cenário 2 (polígonos das árvores refinados)
OUTPUT_POLYGONS_FILENAME_SCENARIO2 = "arvores_cenario2_exg_refinado_poligonos.geojson"

# =============================================================================
# FUNÇÕES DE PROCESSAMENTO
# =============================================================================

def resolve_training_points_path(training_points_path):
    """Tenta resolver o caminho do arquivo de pontos de treino existente."""
    if training_points_path and os.path.exists(training_points_path):
        return training_points_path

    input_dir = os.path.join(os.path.dirname(__file__), "INPUT")
    for filename in DEFAULT_TRAINING_FILENAMES:
        candidate = os.path.join(input_dir, filename)
        if os.path.exists(candidate):
            print(f"Treino não encontrado em '{training_points_path}'. Usando arquivo de treino encontrado: {candidate}")
            return candidate

    return training_points_path


def load_training_points(training_points_path, max_features=None):
    """Carrega pontos de treino usando pyogrio ou geopandas. Se max_features=None, carrega TODOS os pontos."""
    if not training_points_path or not os.path.exists(training_points_path):
        return None

    print(f"Carregando pontos de treino: {training_points_path}")
    gdf = None

    if pyogrio is not None:
        try:
            # Ler TODOS os pontos se max_features for None
            gdf = pyogrio.read_dataframe(training_points_path)
            print(f"Pontos de treino lidos com pyogrio: {len(gdf)} registros")
        except Exception as e:
            print(f"Falha ao ler pontos de treino com pyogrio: {e}")
            gdf = None

    if gdf is None:
        try:
            gdf = gpd.read_file(training_points_path)
            print(f"Pontos de treino lidos com geopandas: {len(gdf)} registros")
        except Exception as e:
            print(f"Falha ao ler pontos de treino com geopandas: {e}")
            return None

    # Se max_features foi especificado, fazer amostragem
    if max_features is not None and len(gdf) > max_features:
        gdf = gdf.sample(n=max_features, random_state=42)
        print(f"Amostragem de treino reduzida para {max_features} pontos")

    return gdf


def calculate_exg(img_rgb):
    """
    Calcula o Índice de Vegetação Excessiva (ExG) para uma imagem RGB.
    ExG = 2 * G - R - B
    """
    r, g, b = img_rgb[:,:,0], img_rgb[:,:,1], img_rgb[:,:,2]
    # Normalizar para 0-1 para cálculo mais estável, depois escalar de volta para 0-255
    r_norm = r / 255.0
    g_norm = g / 255.0
    b_norm = b / 255.0
    exg = 2 * g_norm - r_norm - b_norm
    # Escalar ExG para 0-255 para criar uma imagem de 8 bits
    exg_scaled = cv2.normalize(exg, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    return exg_scaled

def detect_tree_polygons_exg(
    tiff_path,
    output_geojson_path,
    training_points_path=None,
    default_exg_threshold=70, # Limiar padrão para ExG (ajustável)
    min_area=100,
    max_area=8000,
    min_circularity=0.4,
    chunk_size=2000  # Processar em chunks de 2000x2000 pixels para evitar travamento
):
    """
    Detecta polígonos de árvores em uma imagem GeoTIFF usando ExG e os salva como GeoJSON.
    Processa em chunks para imagens grandes. Pode usar pontos de treino para refinar o limiar de ExG.
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
        print(f"Dimensões: {src.shape[0]}x{src.shape[1]} pixels")
        print(f"Processamento em chunks de {chunk_size}x{chunk_size}")

        # 3. Determinar limiar de ExG (Cenário 1 ou 2)
        current_exg_threshold = default_exg_threshold

        if training_points_path and os.path.exists(training_points_path):
            print(f"Cenário 2: Refinando limiar de ExG com pontos de treino de {training_points_path}")
            print("(Lendo amostra de treino para calibração...)")
            try:
                # Carregar uma AMOSTRA dos pontos para calibrar o limiar ExG
                training_gdf = load_training_points(training_points_path, max_features=500)
                if training_gdf is None or training_gdf.empty:
                    print("Aviso: Arquivo de pontos de treino vazio ou não pôde ser lido. Usando limiar ExG padrão.")
                else:
                    exg_values = []
                    # Processar apenas amostra para calibração, não todos os pontos
                    for idx, row in training_gdf.iterrows():
                        if row.geometry.geom_type == 'Point':
                            try:
                                row_px, col_px = src.index(row.geometry.x, row.geometry.y)
                                # Verificar se o ponto está dentro dos limites da imagem
                                if 0 <= row_px < src.shape[0] and 0 <= col_px < src.shape[1]:
                                    # Ler apenas um bloco pequeno ao redor do ponto
                                    window = rasterio.windows.Window(max(0, col_px-5), max(0, row_px-5), 11, 11)
                                    patch = src.read([1, 2, 3], window=window)
                                    patch_rgb = np.moveaxis(patch, 0, -1).astype(np.float64)
                                    # Normalizar para 0-1
                                    patch_rgb = np.clip(patch_rgb / 255.0, 0, 1)
                                    r, g, b = patch_rgb[:,:,0], patch_rgb[:,:,1], patch_rgb[:,:,2]
                                    exg_local = 2*g - r - b
                                    # Filtrar apenas valores válidos (sem NaN ou infinitos)
                                    valid_exg = exg_local[(exg_local > 0.05) & np.isfinite(exg_local)].flatten()
                                    if len(valid_exg) > 2:
                                        exg_values.append(np.median(valid_exg))  # Usar mediana em vez de percentile
                            except Exception as e:
                                continue
                    
                    if exg_values:
                        # Escalar de volta para 0-255
                        current_exg_threshold = int(np.median(exg_values) * 255 * 0.85)
                        current_exg_threshold = max(30, min(200, current_exg_threshold))  # Limitar range sensível
                        print(f"Limiar ExG refinado: {current_exg_threshold} (baseado em amostra)")
                    else:
                        print("Aviso: Não foi possível amostrar ExG dos pontos de treino. Usando limiar ExG padrão.")
            except Exception as e:
                print(f"Erro ao ler pontos de treino ou refinar ExG: {e}. Usando limiar ExG padrão.")
        else:
            print("Cenário 1: Usando limiar ExG padrão.")

        # Processar em chunks para evitar carregar tudo na memória
        polygons = []
        height, width = src.shape

        for row_start in range(0, height, chunk_size):
            row_end = min(row_start + chunk_size, height)
            for col_start in range(0, width, chunk_size):
                col_end = min(col_start + chunk_size, width)
                
                print(f"Processando chunk: linha {row_start}-{row_end}, coluna {col_start}-{col_end}")
                
                # Ler chunk de 3 bandas
                window = rasterio.windows.Window(col_start, row_start, col_end - col_start, row_end - row_start)
                chunk_data = src.read([1, 2, 3], window=window)
                
                chunk_rgb = np.moveaxis(chunk_data, 0, -1).astype(np.float64)
                # Normalizar para 0-1 e cllipar
                chunk_rgb = np.clip(chunk_rgb / 255.0, 0, 1)
                
                chunk_bgr = cv2.cvtColor((chunk_rgb * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
                
                # 1. Pré-processamento: Filtro Bilateral
                blur = cv2.bilateralFilter(chunk_bgr, 9, 75, 75)
                blur_rgb = cv2.cvtColor(blur, cv2.COLOR_BGR2RGB)
                
                # 2. Calcular ExG
                chunk_exg = calculate_exg(blur_rgb)
                
                # 4. Segmentação binária usando o limiar de ExG
                _, mask = cv2.threshold(chunk_exg, current_exg_threshold, 255, cv2.THRESH_BINARY)
                
                # 5. Operações morfológicas
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
                mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
                
                # 6. Encontrar contornos no chunk
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                # 7. Converter contornos para coordenadas geográficas
                for cnt in contours:
                    area = cv2.contourArea(cnt)
                    perimeter = cv2.arcLength(cnt, True)
                    
                    if perimeter == 0:
                        continue
                    
                    circularity = 4 * np.pi * area / (perimeter * perimeter)
                    
                    if min_area < area < max_area and circularity > min_circularity:
                        geo_coords = []
                        for point in cnt.reshape(-1, 2):
                            # Ponto relativo ao chunk + offset do chunk
                            px = col_start + point[0]
                            py = row_start + point[1]
                            lon, lat = rasterio.transform.xy(transform, py, px)
                            geo_coords.append((lon, lat))
                        
                        if len(geo_coords) >= 3:
                            try:
                                polygons.append(Polygon(geo_coords))
                            except Exception as e:
                                pass

        if polygons:
            gdf = gpd.GeoDataFrame(geometry=polygons, crs=crs)
            final_output_path = os.path.join(OUTPUT_DIR, output_geojson_path)
            gdf.to_file(final_output_path, driver='GeoJSON')
            print(f"Detecção concluída: {len(polygons)} polígonos de árvores salvos em {final_output_path}")
        else:
            print("Nenhum polígono de árvore detectado com os parâmetros atuais.")

# =============================================================================
# EXECUÇÃO PRINCIPAL
# =============================================================================

if __name__ == "__main__":
    # Exemplo de uso para Cenário 1 (apenas busca)
    print("\n--- Executando Cenário 1 (Busca Cega com ExG) ---")
    detect_tree_polygons_exg(
        tiff_path=INPUT_IMAGE_PATH,
        output_geojson_path=OUTPUT_POLYGONS_FILENAME_SCENARIO1,
        training_points_path=None # Nulo para Cenário 1
    )

    # Exemplo de uso para Cenário 2 (com pontos de treino)
    resolved_training_path = resolve_training_points_path(TRAINING_POINTS_PATH)
    if resolved_training_path and os.path.exists(resolved_training_path):
        print("\n--- Executando Cenário 2 (Refinamento com Pontos de Treino e ExG) ---")
        detect_tree_polygons_exg(
            tiff_path=INPUT_IMAGE_PATH,
            output_geojson_path=OUTPUT_POLYGONS_FILENAME_SCENARIO2,
            training_points_path=resolved_training_path
        )
    else:
        print("\n--- Cenário 2 (Refinamento com Pontos de Treino) ignorado: arquivo de treino não encontrado em TRAINING_POINTS_PATH nem em INPUT. ---")


