import rasterio
import cv2
import numpy as np
from pathlib import Path
import os
import time
from datetime import timedelta
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import geopandas as gpd
import laspy
from tqdm import tqdm
import warnings
from scipy import ndimage
from shapely.geometry import Point

warnings.filterwarnings("ignore")

# =============================================================================
# CONFIGURACOES GLOBAIS
# =============================================================================
INPUT_IMAGE_PATH = r"D:\TESTES_PYTHON\OPEM_CV_TESTE\imaru\Imaru2.tif"
MDS_PATH = r"D:\TESTES_PYTHON\OPEM_CV_TESTE\imaru\MDS.tif"
BUFFER_SIZE_METERS = 0.85
OUTPUT_DIR = r"D:\TESTES_PYTHON\OPEM_CV_TESTE\v15_"
PROCESS_DIR = os.path.join(OUTPUT_DIR, "process")
FORCE_EPSG = "EPSG:31982"
PIXELS_BUFFER = 1.5 # Buffer em pixels para cada ponto LAS
GROUP_PIXELS = 50 # Número mínimo de pixels para um grupo ser considerado válido

# Config de classes
CLASSIFICATION_CONFIG = {
    "floresta": {
        "shp_path": r"D:\TESTES_PYTHON\OPEM_CV_TESTE\imaru\floresta_pts.shp",
        "output_tif_suffix": "_prob_floresta.tif",
        "output_las_suffix": None,
        "confidence": None,
        "label_value": 1
    },
    "solo": {
        "shp_path": r"D:\TESTES_PYTHON\OPEM_CV_TESTE\imaru\solo_pts.shp",
        "output_tif_suffix": "_prob_solo.tif",
        "output_las_suffix": "_solo_confidence.laz",
        "confidence": 0.00005, # Mantido conforme solicitado
        "label_value": 0
    }
}

# =============================================================================
# FUNCOES AUXILIARES E INDICES
# =============================================================================

def log_message(message, log_file=None, also_print=True):
    msg = f"[{time.strftime("%H:%M:%S")}] {message}"
    if also_print:
        print(msg)
    if log_file:
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(msg + "\n")

def format_time(seconds):
    return str(timedelta(seconds=int(seconds)))

def calculate_indices(img_rgb):
    """Calcula indices de vegetacao e texturas."""
    r = img_rgb[:, :, 0].astype(float)
    g = img_rgb[:, :, 1].astype(float)
    b = img_rgb[:, :, 2].astype(float)
    
    sum_rgb = r + g + b
    sum_rgb[sum_rgb == 0] = 1
    rn, gn, bn = r / sum_rgb, g / sum_rgb, b / sum_rgb
    
    # Indices
    exg = 2 * gn - rn - bn
    exr = 1.4 * rn - gn
    exb = 1.4 * bn - gn
    exgr = exg - exr
    
    # Textura (Variancia Local 3x3)
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    mean = ndimage.uniform_filter(gray.astype(float), size=3)
    sq_mean = ndimage.uniform_filter(gray.astype(float)**2, size=3)
    variance = sq_mean - mean**2
    variance[variance < 0] = 0
    
    # Gradiente Sobel (Bordas)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    sobel = np.sqrt(sobelx**2 + sobely**2)
    
    return {
        "ExG": exg,
        "ExR": exr,
        "ExB": exb,
        "ExGR": exgr,
        "Variance": variance,
        "Sobel": sobel
    }

def calculate_slope(mds, res):
    """Calcula declividade em % a partir do MDS."""
    x, y = np.gradient(mds, res)
    slope = np.sqrt(x**2 + y**2)
    slope_pct = np.tan(np.arctan(slope)) * 100
    return slope_pct

def save_process_raster(data, meta, name, out_dir):
    """Salva rasters intermediarios para validacao."""
    path = os.path.join(out_dir, f"{name}.tif")
    with rasterio.open(path, "w", **meta) as dst:
        dst.write(data.astype(np.float32), 1)
    return path

# =============================================================================
# NOVA ETAPA: FILTRAGEM ESPACIAL DE PONTOS
# =============================================================================
def filter_points_by_spatial_group(
    points_data, # np.vstack de [xs, ys, prob_chunk[rows, cols]]
    gsd, 
    pixels_buffer_val, 
    group_pixels_val, 
    process_dir, 
    log_file,
    class_name
):
    log_message(f"Iniciando filtragem espacial para a classe {class_name}...", log_file)

    if len(points_data) == 0:
        log_message(f"Nenhum ponto para filtrar para a classe {class_name}.", log_file)
        return np.array([]), None

    # 1. Calcular o tamanho do buffer em metros
    buffer_dist_meters = pixels_buffer_val * gsd
    log_message(f"Tamanho do buffer por ponto: {pixels_buffer_val} pixels = {buffer_dist_meters:.2f} metros", log_file)

    # 2. Calcular a área mínima do grupo em metros quadrados
    min_group_area_sq_m = group_pixels_val * (gsd ** 2)
    log_message(f"Área mínima para um grupo de pixels: {group_pixels_val} pixels = {min_group_area_sq_m:.2f} m²", log_file)

    # Criar GeoDataFrame a partir dos pontos
    geometry = [Point(xy) for xy in points_data[:, :2]] # Apenas X e Y
    gdf = gpd.GeoDataFrame(geometry=geometry, crs=FORCE_EPSG)
    log_message(f"Total de pontos para filtragem: {len(gdf)}", log_file)

    # Aplicar buffer e dissolver
    log_message(f"Aplicando buffer de {buffer_dist_meters:.2f}m e dissolvendo polígonos...", log_file)
    buffered_series = gdf.buffer(buffer_dist_meters)
    buffered_gdf = gpd.GeoDataFrame(geometry=buffered_series, crs=FORCE_EPSG)
    dissolved_gdf = buffered_gdf.dissolve()
    log_message(f"Polígonos dissolvidos gerados: {len(dissolved_gdf)}", log_file)

    # Filtrar polígonos por área mínima
    log_message(f"Filtrando polígonos com área menor que {min_group_area_sq_m:.2f} m²...", log_file)
    filtered_polygons = dissolved_gdf[dissolved_gdf.area >= min_group_area_sq_m]
    log_message(f"Polígonos restantes após filtragem: {len(filtered_polygons)}", log_file)

    # Salvar os polígonos filtrados
    output_shp_path = os.path.join(process_dir, f"filtered_groups_{class_name}.shp")
    if not filtered_polygons.empty:
        filtered_polygons.to_file(output_shp_path)
        log_message(f"Polígonos filtrados salvos em: {output_shp_path}", log_file)
    else:
        log_message(f"Nenhum polígono restante para salvar para a classe {class_name}.", log_file)
        output_shp_path = None

    # Filtrar os pontos originais com base nos polígonos resultantes
    log_message("Filtrando pontos originais com base nos polígonos resultantes...", log_file)
    if not filtered_polygons.empty:
        points_in_polygons = gpd.sjoin(gdf, filtered_polygons, how="inner", predicate='within')
        original_indices = points_in_polygons.index.unique()
        filtered_points_data = points_data[original_indices]
        log_message(f"Pontos restantes após filtragem: {len(filtered_points_data)}", log_file)
    else:
        filtered_points_data = np.array([])
        log_message(f"Nenhum ponto restante após filtragem para a classe {class_name}.", log_file)

    return filtered_points_data, output_shp_path

# =============================================================================
# ETAPA 1: EXTRACAO DE FEATURES
# =============================================================================

def extract_features_for_training(tiff_path, mds_path, classification_config, buffer_size_m, log_file, process_dir):
    log_message("=" * 60, log_file)
    log_message("ETAPA 1: EXTRACAO DE FEATURES PARA TREINAMENTO", log_file)
    log_message("=" * 60, log_file)
    
    os.makedirs(process_dir, exist_ok=True)

    with rasterio.open(tiff_path) as src, rasterio.open(mds_path) as src_mds:
        gsd = abs(src.transform[0])
        buffer_px = max(1, int(np.ceil(buffer_size_m / gsd)))
        
        img_rgb = np.moveaxis(src.read([1, 2, 3]), 0, -1)
        mds = src_mds.read(1)
        
        # Gerar indices para a imagem toda (para extrair nos pontos)
        indices = calculate_indices(img_rgb)
        slope = calculate_slope(mds, gsd)
        hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
        
        # Salvar intermediarios para validacao
        meta = src.meta.copy()
        meta.update(dtype="float32", count=1, nodata=np.nan)
        for name, data in indices.items():
            save_process_raster(data, meta, name, process_dir)
        save_process_raster(slope, meta, "Slope", process_dir)
        
        # Lista de nomes de features para o log
        feature_names = ["R", "G", "B", "H", "S", "V", "ExG", "ExR", "ExB", "ExGR", "Variance", "Sobel", "MDS", "Slope"]
        n_features = len(feature_names)

        all_features, all_labels = [], []

        for class_name, config in classification_config.items():
            path = config["shp_path"]
            label_val = config["label_value"]
            
            if Path(path).is_file():
                gdf = gpd.read_file(path)
                log_message(f"Extraindo {len(gdf)} pontos de {class_name}...", log_file)
                
                # Exportar poligonos com buffer para validacao
                buffer_gdf = gdf.copy()
                buffer_gdf.geometry = buffer_gdf.buffer(buffer_size_m)
                buffer_gdf.to_file(os.path.join(process_dir, f"val_buffer_{class_name}.shp"))
                
                f_list, l_list = [], []
                for geom in gdf.geometry:
                    if geom.geom_type == "Point":
                        c, r = src.index(geom.x, geom.y)
                        r_s, r_e = max(0, r - buffer_px), min(src.height, r + buffer_px)
                        c_s, c_e = max(0, c - buffer_px), min(src.width, c + buffer_px)
                        
                        if r_s < r_e and c_s < c_e:
                            # Empilhar todas as features
                            feat = np.hstack([
                                img_rgb[r_s:r_e, c_s:c_e].reshape(-1, 3),
                                hsv[r_s:r_e, c_s:c_e].reshape(-1, 3),
                                indices["ExG"][r_s:r_e, c_s:c_e].reshape(-1, 1),
                                indices["ExR"][r_s:r_e, c_s:c_e].reshape(-1, 1),
                                indices["ExB"][r_s:r_e, c_s:c_e].reshape(-1, 1),
                                indices["ExGR"][r_s:r_e, c_s:c_e].reshape(-1, 1),
                                indices["Variance"][r_s:r_e, c_s:c_e].reshape(-1, 1),
                                indices["Sobel"][r_s:r_e, c_s:c_e].reshape(-1, 1),
                                mds[r_s:r_e, c_s:c_e].reshape(-1, 1),
                                slope[r_s:r_e, c_s:c_e].reshape(-1, 1)
                            ])
                            f_list.append(feat)
                            l_list.append(np.full(feat.shape[0], label_val))
                
                if f_list:
                    all_features.extend(f_list)
                    all_labels.extend(l_list)

    return np.vstack(all_features), np.concatenate(all_labels), feature_names

# =============================================================================
# ETAPA 2: TREINAMENTO
# =============================================================================

def train_model(features, labels, feature_names, log_file):
    log_message("\n" + "=" * 60, log_file)
    log_message("ETAPA 2: TREINAMENTO DO MODELO (Random Forest)", log_file)
    log_message("=" * 60, log_file)
    
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
    
    # Ajuste para ~2k pontos por classe (amostragem se necessario)
    # Como o usuario mencionou 2k pontos, vamos garantir que o modelo lide bem
    
    model = RandomForestClassifier(n_estimators=250, n_jobs=-1, max_depth=25, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    log_message(f"Acuracia: {accuracy_score(y_test, y_pred):.4f}", log_file)
    log_message(f"Relatorio:\n{classification_report(y_test, y_pred)}", log_file)
    
    # Importancia
    importances = sorted(zip(feature_names, model.feature_importances_), key=lambda x: -x[1])
    log_message("\nImportancia das Features:", log_file)
    for name, imp in importances:
        log_message(f"  {name}: {imp:.4f}", log_file)
        
    return model

# =============================================================================
# ETAPA 3: GERACAO RASTER
# =============================================================================

def generate_probability_maps(tiff_path, mds_path, rf_model, classification_config, out_dir, log_file, chunk_size=1024):
    log_message("\n" + "=" * 60, log_file)
    log_message("ETAPA 3: GERACAO DE MAPAS DE PROBABILIDADE E FILTRAGEM DE LAS", log_file)
    log_message("=" * 60, log_file)

    with rasterio.open(tiff_path) as src, rasterio.open(mds_path) as src_mds:
        h, w = src.shape
        gsd = abs(src.transform[0])
        meta = src.meta.copy()
        meta.update(dtype="float32", count=1, nodata=np.nan)
        
        # Mapeamento de indices das classes
        class_indices = {config["label_value"]: np.where(rf_model.classes_ == config["label_value"])[0][0]
                         for config in classification_config.values()}
        
        prob_maps = {name: np.full((h, w), np.nan, dtype=np.float32) for name in classification_config.keys()}
        all_confidence_points = {name: [] for name, cfg in classification_config.items() if cfg["confidence"] is not None}

        total_chunks = (h // chunk_size + 1) * (w // chunk_size + 1)
        with tqdm(total=total_chunks, desc="Processando chunks") as pbar:
            for r_s in range(0, h, chunk_size):
                r_e = min(r_s + chunk_size, h)
                for c_s in range(0, w, chunk_size):
                    c_e = min(c_s + chunk_size, w)
                    
                    win = rasterio.windows.Window(c_s, r_s, c_e - c_s, r_e - r_s)
                    rgb = np.moveaxis(src.read([1, 2, 3], window=win), 0, -1)
                    mds_chunk = src_mds.read(1, window=win)
                    
                    # Mascara Alpha/NoData
                    mask = np.ones((r_e - r_s, c_e - c_s), dtype=bool)
                    if src.count >= 4:
                        alpha = src.read(4, window=win)
                        mask = alpha >= 250
                    
                    if np.any(mask):
                        # Features do chunk
                        indices = calculate_indices(rgb)
                        slope = calculate_slope(mds_chunk, gsd)
                        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
                        
                        feat_valid = np.hstack([
                            rgb[mask].reshape(-1, 3),
                            hsv[mask].reshape(-1, 3),
                            indices["ExG"][mask].reshape(-1, 1),
                            indices["ExR"][mask].reshape(-1, 1),
                            indices["ExB"][mask].reshape(-1, 1),
                            indices["ExGR"][mask].reshape(-1, 1),
                            indices["Variance"][mask].reshape(-1, 1),
                            indices["Sobel"][mask].reshape(-1, 1),
                            mds_chunk[mask].reshape(-1, 1),
                            slope[mask].reshape(-1, 1)
                        ])
                        
                        probas = rf_model.predict_proba(feat_valid)
                        
                        for name, config in classification_config.items():
                            idx = class_indices[config["label_value"]]
                            prob_chunk = np.full((r_e - r_s, c_e - c_s), np.nan, dtype=np.float32)
                            prob_chunk[mask] = probas[:, idx]
                            prob_maps[name][r_s:r_e, c_s:c_e] = prob_chunk
                            
                            # Pontos de Confidencia (LAS)
                            conf_thresh = config.get("confidence")
                            if conf_thresh is not None:
                                rows, cols = np.where((prob_chunk < conf_thresh) & mask)
                                if len(rows) > 0:
                                    xs, ys = rasterio.transform.xy(src.transform, rows + r_s, cols + c_s)
                                    all_confidence_points[name].append(np.vstack([xs, ys, prob_chunk[rows, cols]]).T)
                    pbar.update(1)

        # Salvar Rasters Finais
        for name, config in classification_config.items():
            out_path = os.path.join(out_dir, f"{Path(tiff_path).stem}{config['output_tif_suffix']}")
            with rasterio.open(out_path, "w", **meta) as dst:
                dst.write(prob_maps[name], 1)
            log_message(f"Raster salvo: {out_path}", log_file)

        # Processar e Salvar LAS de Confidencia
        for name, points_list in all_confidence_points.items():
            if points_list:
                all_pts_raw = np.vstack(points_list)
                
                # Aplicar filtragem espacial nos pontos de confiança
                filtered_pts, output_shp_path = filter_points_by_spatial_group(
                    all_pts_raw,
                    gsd,
                    PIXELS_BUFFER,
                    GROUP_PIXELS,
                    PROCESS_DIR,
                    log_file,
                    name
                )

                if len(filtered_pts) > 0:
                    header = laspy.LasHeader(point_format=3, version="1.2")
                    header.add_extra_dims([laspy.ExtraBytesParams(name="confidence", type=np.float32)])
                    las = laspy.LasData(header)
                    las.x, las.y = filtered_pts[:, 0], filtered_pts[:, 1]
                    las.z = np.zeros_like(filtered_pts[:, 0]) # Z-coordinate is not available in prob_chunk, setting to 0 or handle as needed
                    las.confidence = filtered_pts[:, 2]
                    las_path = os.path.join(out_dir, f"{Path(tiff_path).stem}{classification_config[name]['output_las_suffix']}")
                    las.write(las_path)
                    log_message(f"LAS filtrado salvo: {las_path}", log_file)
                else:
                    log_message(f"Nenhum ponto LAS restante após filtragem para a classe {name}. Não foi salvo nenhum arquivo LAS.", log_file)
            else:
                log_message(f"Nenhum ponto de confiança gerado para a classe {name}.", log_file)

if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    log_file = os.path.join(OUTPUT_DIR, "process_log.txt")
    
    try:
        features, labels, f_names = extract_features_for_training(INPUT_IMAGE_PATH, MDS_PATH, CLASSIFICATION_CONFIG, BUFFER_SIZE_METERS, log_file, PROCESS_DIR)
        model = train_model(features, labels, f_names, log_file)
        generate_probability_maps(INPUT_IMAGE_PATH, MDS_PATH, model, CLASSIFICATION_CONFIG, OUTPUT_DIR, log_file)
        log_message("Processo concluído com sucesso.", log_file)
    except Exception as e:
        log_message(f"Erro: {str(e)}", log_file)
        raise
