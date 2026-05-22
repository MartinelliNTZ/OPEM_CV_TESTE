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
from scipy.spatial import KDTree
from shapely.geometry import Point, Polygon, MultiPolygon
from shapely.ops import unary_union
from scipy.ndimage import gaussian_filter

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
GROUP_PIXELS_SHP = GROUP_PIXELS*5 #USADO SOMENTE NO FINAL E APENAS NO FINAL
IDW_K_NEIGHBORS = 24
IDW_POWER = 2.0
MDT_CHUNK_SIZE = 512
MDT_RESOLUTION = 0.50  # resolucao do MDT em metros
IDW_SMOOTH_SIGMA = 1.5  


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
        "confidence": 0.00005,
        "label_value": 0
    }
}

# =============================================================================
# FUNCOES AUXILIARES E INDICES
# =============================================================================

def log_message(message, log_file=None, also_print=True):
    msg = f"[{time.strftime('%H:%M:%S')}] {message}"
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
    
    exg = 2 * gn - rn - bn
    exr = 1.4 * rn - gn
    exb = 1.4 * bn - gn
    exgr = exg - exr
    
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    mean = ndimage.uniform_filter(gray.astype(float), size=3)
    sq_mean = ndimage.uniform_filter(gray.astype(float)**2, size=3)
    variance = sq_mean - mean**2
    variance[variance < 0] = 0
    
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    sobel = np.sqrt(sobelx**2 + sobely**2)
    
    return {
        "ExG": exg, "ExR": exr, "ExB": exb, "ExGR": exgr,
        "Variance": variance, "Sobel": sobel
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
# FILTRAGEM ESPACIAL DE PONTOS (usada dentro da Etapa 3)
# =============================================================================
def filter_points_by_spatial_group(
    points_data,
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

    buffer_dist_meters = pixels_buffer_val * gsd
    log_message(f"Tamanho do buffer por ponto: {pixels_buffer_val} pixels = {buffer_dist_meters:.2f} metros", log_file)

    min_group_area_sq_m = group_pixels_val * (gsd ** 2)
    log_message(f"Área mínima para um grupo de pixels: {group_pixels_val} pixels = {min_group_area_sq_m:.2f} m²", log_file)

    geometry = [Point(xy) for xy in points_data[:, :2]]
    gdf = gpd.GeoDataFrame(geometry=geometry, crs=FORCE_EPSG)
    log_message(f"Total de pontos para filtragem: {len(gdf)}", log_file)

    log_message(f"Aplicando buffer de {buffer_dist_meters:.2f}m e dissolvendo polígonos...", log_file)
    buffered_series = gdf.buffer(buffer_dist_meters)
    log_message(f"Buffers aplicados. Iniciando dissolução...", log_file)
    
    dissolved = unary_union(buffered_series.tolist())
    log_message(f"Dissolução concluída. Tipo geométrico resultante: {dissolved.geom_type}", log_file)
    
    if dissolved.geom_type == 'MultiPolygon':
        polygons = list(dissolved.geoms)
    elif dissolved.geom_type == 'Polygon':
        polygons = [dissolved]
    else:
        polygons = []
    
    dissolved_gdf = gpd.GeoDataFrame(geometry=polygons, crs=FORCE_EPSG)
    log_message(f"Polígonos dissolvidos gerados: {len(dissolved_gdf)}", log_file)

    log_message(f"Filtrando polígonos com área menor que {min_group_area_sq_m:.2f} m²...", log_file)
    filtered_polygons = dissolved_gdf[dissolved_gdf.area >= min_group_area_sq_m]
    log_message(f"Polígonos restantes após filtragem: {len(filtered_polygons)}", log_file)

    output_shp_path = os.path.join(process_dir, f"filtered_groups_{class_name}.shp")
    if not filtered_polygons.empty:
        filtered_polygons.to_file(output_shp_path)
        log_message(f"Polígonos filtrados salvos em: {output_shp_path}", log_file)
    else:
        log_message(f"Nenhum polígono restante para salvar para a classe {class_name}.", log_file)
        output_shp_path = None

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
        
        indices = calculate_indices(img_rgb)
        slope = calculate_slope(mds, gsd)
        hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
        
        meta = src.meta.copy()
        meta.update(dtype="float32", count=1, nodata=np.nan)
        for name, data in indices.items():
            save_process_raster(data, meta, name, process_dir)
        save_process_raster(slope, meta, "Slope", process_dir)
        
        feature_names = ["R", "G", "B", "H", "S", "V", "ExG", "ExR", "ExB", "ExGR", "Variance", "Sobel", "MDS", "Slope"]

        all_features, all_labels = [], []

        for class_name, config in classification_config.items():
            path = config["shp_path"]
            label_val = config["label_value"]
            
            if Path(path).is_file():
                gdf = gpd.read_file(path)
                log_message(f"Extraindo {len(gdf)} pontos de {class_name}...", log_file)
                
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
    
    model = RandomForestClassifier(n_estimators=250, n_jobs=-1, max_depth=25, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    log_message(f"Acuracia: {accuracy_score(y_test, y_pred):.4f}", log_file)
    log_message(f"Relatorio:\n{classification_report(y_test, y_pred)}", log_file)
    
    importances = sorted(zip(feature_names, model.feature_importances_), key=lambda x: -x[1])
    log_message("\nImportancia das Features:", log_file)
    for name, imp in importances:
        log_message(f"  {name}: {imp:.4f}", log_file)
        
    return model

# =============================================================================
# ETAPA 3: GERACAO RASTER E LAS DE CONFIDENCIA
# =============================================================================

def generate_probability_maps(tiff_path, mds_path, rf_model, classification_config, out_dir, log_file, chunk_size=1024):
    log_message("\n" + "=" * 60, log_file)
    log_message("ETAPA 3: GERACAO DE MAPAS DE PROBABILIDADE E FILTRAGEM DE LAS", log_file)
    log_message("=" * 60, log_file)

    las_path_result = None
    shape_paths = {}

    with rasterio.open(tiff_path) as src, rasterio.open(mds_path) as src_mds:
        h, w = src.shape
        gsd = abs(src.transform[0])
        meta = src.meta.copy()
        meta.update(dtype="float32", count=1, nodata=np.nan)
        
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
                    
                    mask = np.ones((r_e - r_s, c_e - c_s), dtype=bool)
                    if src.count >= 4:
                        alpha = src.read(4, window=win)
                        mask = alpha >= 250
                    
                    if np.any(mask):
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
                
                filtered_pts, output_shp_path = filter_points_by_spatial_group(
                    all_pts_raw, gsd, PIXELS_BUFFER, GROUP_PIXELS,
                    PROCESS_DIR, log_file, name
                )
                
                shape_paths[name] = output_shp_path

                if len(filtered_pts) > 0:
                    header = laspy.LasHeader(point_format=3, version="1.2")
                    header.add_extra_dims([laspy.ExtraBytesParams(name="confidence", type=np.float32)])
                    las = laspy.LasData(header)
                    las.x, las.y = filtered_pts[:, 0], filtered_pts[:, 1]
                    las.z = np.zeros_like(filtered_pts[:, 0])
                    las.confidence = filtered_pts[:, 2]
                    las_path = os.path.join(out_dir, f"{Path(tiff_path).stem}{classification_config[name]['output_las_suffix']}")
                    las.write(las_path)
                    log_message(f"LAS filtrado salvo: {las_path}", log_file)
                    las_path_result = las_path
                else:
                    log_message(f"Nenhum ponto LAS restante após filtragem para a classe {name}.", log_file)
            else:
                log_message(f"Nenhum ponto de confiança gerado para a classe {name}.", log_file)

    return las_path_result, shape_paths


# =============================================================================
# ETAPA 4: EXTRACAO DE Z DO MDS
# =============================================================================

def extract_z_from_mds(las_path, mds_path, log_file):
    log_message("\n" + "=" * 60, log_file)
    log_message("ETAPA 4: EXTRACAO DE Z DO MDS PARA OS PONTOS LAS", log_file)
    log_message("=" * 60, log_file)
    t0 = time.time()

    log_message(f"   -> LAS: {las_path}", log_file)
    las_data = laspy.read(las_path)
    n_points = len(las_data)
    log_message(f"   -> Pontos: {n_points:,}", log_file)

    xs = las_data.x.copy()
    ys = las_data.y.copy()
    log_message(f"   -> Coordenadas X/Y carregadas: {len(xs):,}", log_file)

    with rasterio.open(mds_path) as mds_src:
        log_message(f"   -> MDS: {mds_path}", log_file)
        log_message(f"   -> MDS shape: {mds_src.shape}", log_file)
        log_message(f"   -> MDS CRS: {mds_src.crs}", log_file)
        log_message(f"   -> MDS resolucao: {mds_src.res[0]:.4f}m", log_file)

        transform = mds_src.transform
        cols = np.floor((xs - transform.c) / transform.a).astype(np.int64)
        rows = np.floor((transform.f - ys) / (-transform.e)).astype(np.int64)

        valid_mask = (rows >= 0) & (rows < mds_src.height) & (cols >= 0) & (cols < mds_src.width)
        n_valid = np.sum(valid_mask)
        n_out = len(xs) - n_valid
        log_message(f"   -> Pontos dentro do MDS: {n_valid:,}", log_file)
        log_message(f"   -> Pontos fora do MDS: {n_out:,}", log_file)

        z_values = np.full(len(xs), np.nan, dtype=np.float32)
        row_min, row_max = int(rows[valid_mask].min()), int(rows[valid_mask].max())

        for r0 in range(row_min, row_max + 1, 4096):
            r1 = min(r0 + 4096, row_max + 1)
            mds_block = mds_src.read(1, window=rasterio.windows.Window(0, r0, mds_src.width, r1 - r0))

            in_block = valid_mask & (rows >= r0) & (rows < r1)
            idx_block = np.where(in_block)[0]
            if len(idx_block) > 0:
                local_rows = rows[idx_block] - r0
                local_cols = cols[idx_block]
                z_block = mds_block[local_rows, local_cols]
                nodata = mds_src.nodata
                if nodata is not None:
                    z_block = np.where(z_block == nodata, np.nan, z_block)
                z_values[idx_block] = z_block

        z_valid_mask = ~np.isnan(z_values)
        z_final = z_values[z_valid_mask]
        x_final = xs[z_valid_mask]
        y_final = ys[z_valid_mask]

        log_message(f"   -> Z extraidos com sucesso: {len(z_final):,}", log_file)
        log_message(f"   -> Z min: {np.nanmin(z_final):.2f}m", log_file)
        log_message(f"   -> Z max: {np.nanmax(z_final):.2f}m", log_file)
        log_message(f"   -> Z medio: {np.nanmean(z_final):.2f}m", log_file)
        log_message(f"   -> Tempo: {format_time(time.time() - t0)}", log_file)

    return x_final, y_final, z_final


# =============================================================================
# ETAPA 5: GERAR MDT POR IDW + KDTREE
# =============================================================================

# pip install pykrige
# =============================================================================
# CONFIGURAÇÕES
# =============================================================================
KRIG_MODEL      = 'spherical'   # 'spherical', 'exponential', 'gaussian', 'linear'
KRIG_NLAGS      = 20            # lags para ajuste automático do variograma
KRIG_MAX_POINTS = 50_000        # subsample se tiver muitos pontos (velocidade)
MDT_CHUNK_SIZE  = 256           # chunks menores para krigagem (mais RAM por chunk)

# =============================================================================
# ETAPA 5: GERAR MDT POR KRIGAGEM ORDINÁRIA AUTOMÁTICA
# =============================================================================
from pykrige.ok import OrdinaryKriging
import numpy as np
import rasterio
from tqdm import tqdm
import os, time

def generate_mdt_kriging(xs, ys, zs, bounds, resolution, epsg, output_dir, base_name, log_file):
    log_message("\n" + "=" * 60, log_file)
    log_message("ETAPA 5: GERACAO DO MDT POR KRIGAGEM ORDINARIA", log_file)
    log_message("=" * 60, log_file)
    t0 = time.time()

    left, bottom, right, top = bounds
    width  = int(np.ceil((right - left) / resolution))
    height = int(np.ceil((top  - bottom) / resolution))

    log_message(f"   -> Dimensoes raster: {width} x {height} pixels", log_file)
    log_message(f"   -> Pontos de entrada: {len(xs):,}", log_file)

    # ------------------------------------------------------------------
    # 1. Subsample se necessário (Kriging é O(n³) no ajuste)
    # ------------------------------------------------------------------
    if len(xs) > KRIG_MAX_POINTS:
        log_message(f"   -> Subsampling para {KRIG_MAX_POINTS:,} pontos...", log_file)
        idx = np.random.choice(len(xs), KRIG_MAX_POINTS, replace=False)
        xs_k, ys_k, zs_k = xs[idx], ys[idx], zs[idx]
    else:
        xs_k, ys_k, zs_k = xs, ys, zs

    # ------------------------------------------------------------------
    # 2. Ajuste automático do variograma (sem análise manual)
    # ------------------------------------------------------------------
    log_message(f"\n   Ajustando variograma automatico (modelo={KRIG_MODEL})...", log_file)
    t_vario = time.time()

    OK = OrdinaryKriging(
        xs_k, ys_k, zs_k,
        variogram_model=KRIG_MODEL,   # ajuste automático dos parâmetros
        nlags=KRIG_NLAGS,
        weight=True,                  # pondera lags por número de pares
        enable_plotting=False,
        verbose=False,
    )

    log_message(f"   -> Variograma ajustado em {format_time(time.time() - t_vario)}", log_file)
    log_message(f"   -> Nugget : {OK.variogram_model_parameters[2]:.4f}", log_file)
    log_message(f"   -> Sill   : {OK.variogram_model_parameters[0]:.4f}", log_file)
    log_message(f"   -> Range  : {OK.variogram_model_parameters[1]:.4f}m", log_file)

    # ------------------------------------------------------------------
    # 3. Predição por chunks
    # ------------------------------------------------------------------
    transform = rasterio.transform.from_origin(left, top, resolution, resolution)
    meta = {
        "driver": "GTiff", "dtype": "float32", "nodata": np.nan,
        "width": width, "height": height, "count": 1, "crs": epsg,
        "transform": transform, "compress": "lzw", "tiled": True,
        "blockxsize": MDT_CHUNK_SIZE, "blockysize": MDT_CHUNK_SIZE,
    }

    output_path = os.path.join(output_dir, f"{base_name}.tif")

    num_chunks_h = int(np.ceil(height / MDT_CHUNK_SIZE))
    num_chunks_w = int(np.ceil(width  / MDT_CHUNK_SIZE))
    total_chunks = num_chunks_h * num_chunks_w

    log_message(f"\n   Interpolando por chunks ({total_chunks} chunks)...", log_file)

    with rasterio.open(output_path, "w", **meta) as dst:
        with tqdm(total=total_chunks, desc="Kriging chunks", unit="chunk") as pbar:
            for r0 in range(0, height, MDT_CHUNK_SIZE):
                r1 = min(r0 + MDT_CHUNK_SIZE, height)
                for c0 in range(0, width, MDT_CHUNK_SIZE):
                    c1 = min(c0 + MDT_CHUNK_SIZE, width)

                    chunk_h = r1 - r0
                    chunk_w = c1 - c0

                    pixel_x = left + (c0 + np.arange(chunk_w) + 0.5) * resolution
                    pixel_y = top  - (r0 + np.arange(chunk_h) + 0.5) * resolution

                    # pykrige recebe vetores 1D de x e y (grade regular)
                    z_pred, _ = OK.execute(
                        'grid',
                        pixel_x,
                        pixel_y,
                        backend='vectorized',   # 'loop' usa menos RAM se necessário
                    )

                    dst.write(z_pred.astype(np.float32), 1,
                              window=rasterio.windows.Window(c0, r0, chunk_w, chunk_h))
                    pbar.update(1)

    t_total = time.time() - t0
    log_message(f"\n   -> MDT gerado em: {output_path}", log_file)
    log_message(f"   -> Tempo total Kriging: {format_time(t_total)}", log_file)

    with rasterio.open(output_path) as result:
        data = result.read(1)
        valid_data = data[~np.isnan(data)]
        if len(valid_data) > 0:
            log_message(f"   -> MDT Z min:   {np.nanmin(valid_data):.2f}m", log_file)
            log_message(f"   -> MDT Z max:   {np.nanmax(valid_data):.2f}m", log_file)
            log_message(f"   -> MDT Z medio: {np.nanmean(valid_data):.2f}m", log_file)
            log_message(f"   -> MDT STD:     {np.nanstd(valid_data):.2f}m", log_file)

    return output_path


# =============================================================================
# ETAPA 6: FILTRAGEM AGRESSIVA DO SHAPE (POS-MDT)
# =============================================================================

def filter_shape_aggressive(shp_path, gsd, process_dir, log_file):
    """Remove buracos pequenos e features muito grandes do shape filtrado."""
    log_message("\n" + "=" * 60, log_file)
    log_message("ETAPA 6: FILTRAGEM AGRESSIVA DO SHAPE (POS-MDT)", log_file)
    log_message("=" * 60, log_file)
    t0 = time.time()

    # Carregar shape
    gdf = gpd.read_file(shp_path)
    log_message(f"Shape carregado: {shp_path}", log_file)
    log_message(f"Total de features: {len(gdf)}", log_file)

    # Thresholds
    min_hole_area_sq_m = GROUP_PIXELS_SHP * (gsd ** 2)       # GROUP_PIXELS * 5 * gsd²
    max_feature_area_sq_m = GROUP_PIXELS_SHP * 5 * (gsd ** 2)  # GROUP_PIXELS * 25 * gsd²

    log_message(f"GSD: {gsd:.4f}m", log_file)
    log_message(f"GROUP_PIXELS_SHP: {GROUP_PIXELS_SHP} pixels = {GROUP_PIXELS} * 5", log_file)
    log_message(f"Area minima para buracos (GROUP_PIXELS_SHP * gsd²): {min_hole_area_sq_m:.2f} m²", log_file)
    log_message(f"Area maxima para feicoes (GROUP_PIXELS_SHP * 5 * gsd²): {max_feature_area_sq_m:.2f} m²", log_file)

    # Remover buracos pequenos de cada poligono
    cleaned_geometries = []
    total_holes_removed = 0

    for idx, row in gdf.iterrows():
        geom = row.geometry
        
        if geom.geom_type == 'Polygon':
            # Exterior ring
            exterior = geom.exterior
            # Filtrar interior rings (buracos) por area
            new_interiors = []
            for interior in geom.interiors:
                hole_poly = Polygon(interior)
                if hole_poly.area >= min_hole_area_sq_m:
                    new_interiors.append(interior)
                else:
                    total_holes_removed += 1
            
            # Recriar poligono sem buracos pequenos
            cleaned_geom = Polygon(exterior, new_interiors) if new_interiors else Polygon(exterior)
            cleaned_geometries.append(cleaned_geom)
            
        elif geom.geom_type == 'MultiPolygon':
            # Processar cada sub-poligono
            cleaned_sub_polys = []
            for sub_poly in geom.geoms:
                exterior = sub_poly.exterior
                new_interiors = []
                for interior in sub_poly.interiors:
                    hole_poly = Polygon(interior)
                    if hole_poly.area >= min_hole_area_sq_m:
                        new_interiors.append(interior)
                    else:
                        total_holes_removed += 1
                
                cleaned_sub = Polygon(exterior, new_interiors) if new_interiors else Polygon(exterior)
                cleaned_sub_polys.append(cleaned_sub)
            
            # MultiPolygon ou Polygon unico
            if len(cleaned_sub_polys) > 1:
                cleaned_geometries.append(MultiPolygon(cleaned_sub_polys))
            else:
                cleaned_geometries.append(cleaned_sub_polys[0])
        else:
            cleaned_geometries.append(geom)

    log_message(f"Total de buracos removidos: {total_holes_removed}", log_file)

    # Criar GeoDataFrame com geometrias limpas
    gdf_cleaned = gpd.GeoDataFrame(geometry=cleaned_geometries, crs=gdf.crs)
    
    # Calcular area como atributo
    gdf_cleaned['area_m2'] = gdf_cleaned.area
    
    # Remover features com area 0 (poligonos degenerados)
    gdf_cleaned = gdf_cleaned[gdf_cleaned['area_m2'] > 0].copy()
    
    log_message(f"Features apos remocao de buracos: {len(gdf_cleaned)}", log_file)
    if len(gdf_cleaned) > 0:
        log_message(f"Areas (min/mean/max): {gdf_cleaned['area_m2'].min():.2f} / "
                    f"{gdf_cleaned['area_m2'].mean():.2f} / {gdf_cleaned['area_m2'].max():.2f} m²", log_file)

    # Aplicar filtro: remover feicoes com area > max_feature_area_sq_m (5x maiores que GROUP_PIXELS_SHP)
    gdf_filtered = gdf_cleaned[gdf_cleaned['area_m2'] <= max_feature_area_sq_m].copy()
    n_removed_area = len(gdf_cleaned) - len(gdf_filtered)

    log_message(f"Features apos filtro de area maxima: {len(gdf_filtered)} (removidas {n_removed_area} feicoes "
                f"com area > {max_feature_area_sq_m:.2f} m²)", log_file)

    if not gdf_filtered.empty:
        log_message(f"Areas finais (min/mean/max): {gdf_filtered['area_m2'].min():.2f} / "
                    f"{gdf_filtered['area_m2'].mean():.2f} / {gdf_filtered['area_m2'].max():.2f} m²", log_file)

    # Salvar shape filtrado
    output_shp_path = os.path.join(process_dir, "filtered_groups_aggressive.shp")
    if not gdf_filtered.empty:
        gdf_filtered.to_file(output_shp_path)
        log_message(f"Shape filtrado salvo em: {output_shp_path}", log_file)
    else:
        log_message("Nenhuma feature restante apos filtragem agressiva.", log_file)
        output_shp_path = None

    # Salvar shape com buracos removidos (antes do filtro de area maxima) para debug
    gdf_no_holes_path = os.path.join(process_dir, "filtered_groups_no_holes.shp")
    if not gdf_cleaned.empty:
        gdf_cleaned.to_file(gdf_no_holes_path)
        log_message(f"Shape sem buracos pequenos salvo em: {gdf_no_holes_path}", log_file)

    log_message(f"Tempo Etapa 6: {format_time(time.time() - t0)}", log_file)

    return output_shp_path


# =============================================================================
# PIPELINE PRINCIPAL
# =============================================================================

if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    log_file = os.path.join(OUTPUT_DIR, "process_log.txt")
    
    try:
        # ETAPA 1: Feature extraction
        features, labels, f_names = extract_features_for_training(
            INPUT_IMAGE_PATH, MDS_PATH, CLASSIFICATION_CONFIG, 
            BUFFER_SIZE_METERS, log_file, PROCESS_DIR
        )
        
        # ETAPA 2: Treinamento
        model = train_model(features, labels, f_names, log_file)
        
        # ETAPA 3: Mapas de probabilidade + LAS filtrado + Shape
        las_path, shape_paths = generate_probability_maps(
            INPUT_IMAGE_PATH, MDS_PATH, model, CLASSIFICATION_CONFIG, 
            OUTPUT_DIR, log_file
        )
        
        if las_path is not None:
            # ETAPA 4: Extrair Z do MDS para os pontos LAS
            xs, ys, zs = extract_z_from_mds(las_path, MDS_PATH, log_file)
            
            # ETAPA 5: Gerar MDT por IDW
            # Usar bounds do LAS para definir a extensao do MDT
            bounds = (xs.min(), ys.min(), xs.max(), ys.max())
            mdt_path = generate_mdt_kriging(
                xs, ys, zs, bounds, MDT_RESOLUTION, FORCE_EPSG,
                OUTPUT_DIR, f"{Path(INPUT_IMAGE_PATH).stem}_MDT_KRIGING", log_file
            )
            
            # ETAPA 6: Filtragem agressiva do shape (pos-MDT)
            # Usar o shape gerado na etapa 3 (classe "solo")
            gsd = abs(rasterio.open(INPUT_IMAGE_PATH).transform[0])
            
            # Encontrar shape path da classe que gerou pontos
            shape_filtered_path = None
            for class_name, shp in shape_paths.items():
                if shp is not None:
                    shape_filtered_path = shp
                    log_message(f"Usando shape da classe '{class_name}' para Etapa 6: {shp}", log_file)
                    break
            
            if shape_filtered_path is not None and os.path.exists(shape_filtered_path):
                aggressive_shp_path = filter_shape_aggressive(
                    shape_filtered_path, gsd, PROCESS_DIR, log_file
                )
                log_message(f"\nPipeline completo! MDT: {mdt_path}", log_file)
                log_message(f"Shape filtrado (agressivo): {aggressive_shp_path}", log_file)
            else:
                log_message("Nenhum shape disponivel para Etapa 6.", log_file)
                log_message(f"\nPipeline completo! MDT gerado: {mdt_path}", log_file)
        else:
            log_message("Nenhum LAS gerado na Etapa 3. Pulando Etapas 4, 5 e 6.", log_file)
        
        log_message("Processo concluído com sucesso.", log_file)
        
    except Exception as e:
        log_message(f"Erro: {str(e)}", log_file)
        raise