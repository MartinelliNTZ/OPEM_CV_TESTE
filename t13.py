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
from scipy.spatial import KDTree
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

# =============================================================================
# CONFIGURACOES GLOBAIS
# =============================================================================
INPUT_IMAGE_PATH = r"D:\TESTES_PYTHON\OPEM_CV_TESTE\imaru\Imaru2.tif"
MDS_PATH = r"D:\TESTES_PYTHON\OPEM_CV_TESTE\imaru\MDS.tif"
BUFFER_SIZE_METERS = 0.85
OUTPUT_DIR = r"D:\TESTES_PYTHON\OPEM_CV_TESTE\v13"
FORCE_EPSG = "EPSG:31982"

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
        "confidence": 0.00001,
        "label_value": 0
    }
}

# Parametros IDW para MDT
IDW_K_NEIGHBORS = 12
IDW_POWER = 2.0
MDT_RESOLUTION = 1.0       # metros
MDT_CHUNK_SIZE = 256

# =============================================================================
# FUNCOES AUXILIARES
# =============================================================================

def log_message(message, log_file=None, also_print=True):
    msg = f"[{time.strftime('%H:%M:%S')}] {message}"
    if also_print:
        try:
            print(msg)
        except UnicodeEncodeError:
            sanitized = msg.encode("ascii", errors="replace").decode("ascii")
            print(sanitized)
    if log_file:
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(msg + "\n")


def format_time(seconds):
    return str(timedelta(seconds=int(seconds)))


def calculate_exg(img_rgb):
    r, g, b = (
        img_rgb[:, :, 0].astype(float),
        img_rgb[:, :, 1].astype(float),
        img_rgb[:, :, 2].astype(float),
    )
    sum_rgb = r + g + b
    sum_rgb[sum_rgb == 0] = 1
    r_n, g_n, b_n = r / sum_rgb, g / sum_rgb, b / sum_rgb
    return 2 * g_n - r_n - b_n


def get_epsg_from_las(las_path):
    """Tenta extrair EPSG do cabeçalho VLR do LAS."""
    try:
        with laspy.open(las_path) as las:
            for vlr in las.header.vlrs:
                if hasattr(vlr, 'record_id') and vlr.record_id == 2111:
                    wkt = vlr.parsed_string
                    if "EPSG" in wkt or "epsg" in wkt or "Epsg" in wkt:
                        import re
                        match = re.search(r'EPSG["\s]*[\[\],]*\s*(\d+)', wkt, re.IGNORECASE)
                        if match:
                            epsg_code = match.group(1).strip()
                            log_message(f"[LAS] EPSG: {epsg_code} extraido do cabeçalho LAS.")
                            return f"EPSG:{epsg_code}"
                if hasattr(vlr, 'record_id') and vlr.record_id == 34735:
                    geo_data = vlr.parsed_bytes
                    try:
                        geo_keys = np.frombuffer(geo_data, dtype=np.uint16).reshape(-1, 4)
                        for key in geo_keys:
                            if key[0] == 3072:
                                epsg_code = key[3]
                                if epsg_code > 0:
                                    log_message(f"[LAS] EPSG: {epsg_code} extraido de GeoTIFF VLR.")
                                    return f"EPSG:{epsg_code}"
                    except:
                        pass
    except Exception:
        pass
    return None


# =============================================================================
# ETAPA 1: EXTRACAO DE FEATURES
# =============================================================================

def extract_features_for_training(tiff_path, classification_config, buffer_size_m, log_file):
    log_message("=" * 60, log_file)
    log_message("ETAPA 1: EXTRACAO DE FEATURES PARA TREINAMENTO", log_file)
    log_message("=" * 60, log_file)
    log_message(f"Imagem de entrada: {tiff_path}", log_file)

    with rasterio.open(tiff_path) as src:
        gsd = abs(src.transform[0])
        buffer_px = max(1, int(np.ceil(buffer_size_m / gsd)))

        h_img, w_img = src.shape
        n_bands = src.count
        crs_str = str(src.crs) if src.crs else "Nao definido"
        bounds = src.bounds
        log_message(f"Dimensoes da imagem: {w_img} x {h_img} pixels ({n_bands} bandas)", log_file)
        log_message(f"Resolucao (GSD): {gsd:.6f} m/pixel", log_file)
        log_message(f"CRS: {crs_str}", log_file)
        log_message(f"Extent (bounds): left={bounds.left:.2f}, bottom={bounds.bottom:.2f}, right={bounds.right:.2f}, top={bounds.top:.2f}", log_file)
        log_message(f"Buffer de treinamento: {buffer_size_m}m -> {buffer_px} pixels", log_file)
        log_message(f"Usando bandas 1(R), 2(G), 3(B) para features", log_file)

        img_rgb = np.moveaxis(src.read([1, 2, 3]), 0, -1)
        img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
        img_exg = calculate_exg(img_rgb)
        n_features = 7

        all_features, all_labels = [], []

        def process_class(path, label_val, name, src_obj, rgb, hsv, exg, b_px):
            if Path(path).is_file():
                gdf = gpd.read_file(path)
                log_message(f"Extraindo {len(gdf)} pontos de {name}...", log_file)
                f_list, l_list = [], []
                valid_count = 0
                out_of_bounds = 0
                for geom in gdf.geometry:
                    if geom.geom_type == "Point":
                        c, r = src_obj.index(geom.x, geom.y)
                        r_s, r_e = max(0, r - b_px), min(src_obj.height, r + b_px)
                        c_s, c_e = max(0, c - b_px), min(src_obj.width, c + b_px)
                        if r_s < r_e and c_s < c_e:
                            feat = np.hstack([rgb[r_s:r_e, c_s:c_e].reshape(-1, 3),
                                              hsv[r_s:r_e, c_s:c_e].reshape(-1, 3),
                                              exg[r_s:r_e, c_s:c_e].reshape(-1, 1)])
                            f_list.append(feat)
                            l_list.append(np.full(feat.shape[0], label_val))
                            valid_count += 1
                        else:
                            out_of_bounds += 1
                log_message(f"  -> {name}: {valid_count} pontos dentro da imagem, {out_of_bounds} fora dos limites", log_file)
                return f_list, l_list
            else:
                log_message(f"  -> Arquivo de {name} nao encontrado: {path}", log_file)
                return [], []

        for class_name, config in classification_config.items():
            f_class, l_class = process_class(config["shp_path"], config["label_value"], class_name, src, img_rgb, img_hsv, img_exg, buffer_px)
            all_features.extend(f_class)
            all_labels.extend(l_class)

    if not all_features:
        log_message("ERRO: Nenhuma feature extraida! Verifique os arquivos de pontos.", log_file)
        return np.empty((0, n_features)), np.empty((0,))

    total_pixels = sum(f.shape[0] for f in all_features)
    log_message(f"\nResumo da extracao:", log_file)
    log_message(f"  -> Features por pixel: R(1) G(1) B(1) | H(1) S(1) V(1) | ExG(1) = {n_features} features", log_file)
    for class_name, config in classification_config.items():
        class_pixels = sum(np.sum(l == config["label_value"]) for l in all_labels if len(l) > 0)
        log_message(f"  -> Total de pixels de {class_name} para treino: {class_pixels:,}", log_file)
    log_message(f"  -> Total de amostras (pixels): {total_pixels:,}", log_file)

    return np.vstack(all_features), np.concatenate(all_labels)


# =============================================================================
# ETAPA 2: TREINAMENTO
# =============================================================================

def train_model(features, labels, log_file):
    log_message("\n" + "=" * 60, log_file)
    log_message("ETAPA 2: TREINAMENTO DO MODELO (Random Forest)", log_file)
    log_message("=" * 60, log_file)
    t0 = time.time()

    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
    log_message(f"Tamanho do conjunto de treino: {X_train.shape[0]:,} pixels", log_file)
    log_message(f"Tamanho do conjunto de teste: {X_test.shape[0]:,} pixels", log_file)

    for label_val in np.unique(labels):
        log_message(f"Distribuicao treino - Classe {label_val}: {np.sum(y_train==label_val)}", log_file)
        log_message(f"Distribuicao teste  - Classe {label_val}: {np.sum(y_test==label_val)}", log_file)

    log_message("Treinando Random Forest (n_estimators=100, max_depth=20)...", log_file)
    model = RandomForestClassifier(n_estimators=250, n_jobs=-1, max_depth=20, random_state=42)
    model.fit(X_train, y_train)
    train_time = time.time() - t0

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    log_message(f"\nTempo de treinamento: {format_time(train_time)}", log_file)
    log_message(f"Acuracia: {acc:.4f}", log_file)
    log_message(f"Relatorio de Classificacao:\n{report}", log_file)

    feature_names = ["R", "G", "B", "H", "S", "V", "ExG"]
    log_message(f"\nImportancia das Features (Random Forest):", log_file)
    importances = sorted(zip(feature_names, model.feature_importances_), key=lambda x: -x[1])
    for name, imp in importances:
        log_message(f"  {name}: {imp:.4f} ({imp*100:.1f}%)", log_file)

    log_message(f"\nClasses do modelo (ordem predict_proba): {model.classes_}", log_file)
    log_message(f"Numero de classes: {len(model.classes_)}", log_file)
    return model


# =============================================================================
# ETAPA 3: CLASSIFICACAO + GERACAO MAPAS/LAS
# =============================================================================

def generate_probability_maps(tiff_path, rf_model, classification_config, out_dir, log_file,
                                chunk_size=1024, batch_size=200000):
    log_message("\n" + "=" * 60, log_file)
    log_message("ETAPA 3: GERACAO DE MAPAS DE PROBABILIDADE E PONTOS DE CONFIDENCIA", log_file)
    log_message("=" * 60, log_file)

    input_path = Path(tiff_path)
    output_base_name = input_path.stem
    output_dir = Path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    log_message(f"\nConfiguracoes de Geracao:", log_file)
    log_message(f"  -> Imagem de entrada: {tiff_path}", log_file)
    log_message(f"  -> Diretorio de saida: {output_dir}", log_file)
    log_message(f"  -> Nome base dos arquivos de saida: {output_base_name}", log_file)
    log_message(f"  -> Tamanho do chunk: {chunk_size}x{chunk_size} pixels", log_file)
    log_message(f"  -> Tamanho do batch de predicao: {batch_size:,} pixels", log_file)

    t_pipeline_start = time.time()

    class_indices = {config["label_value"]: np.where(rf_model.classes_ == config["label_value"])[0][0]
                     for class_name, config in classification_config.items()}

    log_message(f"\n[INFO] Mapeamento de classes: rf_model.classes_ = {rf_model.classes_}", log_file)
    for class_name, config in classification_config.items():
        log_message(f"[INFO] {class_name}_class_idx = {class_indices[config['label_value']]} "
                    f"(probabilidade de {class_name}, classe {config['label_value']})", log_file)

    with rasterio.open(tiff_path) as src:
        h, w = src.shape
        meta = src.meta.copy()
        meta.update(dtype="float32", count=1, nodata=np.nan)

        # Verificar NoData / banda alpha
        nodata_val = src.nodata
        has_alpha = False
        alpha_band_idx = None
        for i in range(1, src.count + 1):
            if src.colorinterp[i-1] == rasterio.enums.ColorInterp.alpha:
                has_alpha = True
                alpha_band_idx = i
                break

        if has_alpha:
            log_message(f"[INFO] Banda Alpha detectada (banda {alpha_band_idx}). Usando Alpha < 250 como NoData.", log_file)
        elif nodata_val is not None:
            log_message(f"[INFO] Valor NoData detectado: {nodata_val}", log_file)
        else:
            log_message("[INFO] Nenhum NoData ou Banda Alpha detectado.", log_file)

        prob_maps = {class_name: np.full((h, w), np.nan, dtype=np.float32)
                     for class_name in classification_config.keys()}

        all_confidence_points = {class_name: [] for class_name in classification_config.keys()
                                 if classification_config[class_name]["confidence"] is not None}

        log_message(f"\nClassificando a imagem inteira com Random Forest...", log_file)
        log_message(f"  -> Dimensoes: {w}x{h} pixels = {w*h:,} pixels", log_file)

        total_chunks = (h // chunk_size + (1 if h % chunk_size > 0 else 0)) * \
                       (w // chunk_size + (1 if w % chunk_size > 0 else 0))

        with tqdm(total=total_chunks, desc="Processando chunks", unit="chunk") as pbar:
            for r_s in range(0, h, chunk_size):
                r_e = min(r_s + chunk_size, h)
                for c_s in range(0, w, chunk_size):
                    c_e = min(c_s + chunk_size, w)

                    win = rasterio.windows.Window(c_s, r_s, c_e - c_s, r_e - r_s)
                    rgb = np.moveaxis(src.read([1, 2, 3], window=win), 0, -1)

                    if has_alpha:
                        alpha = src.read(alpha_band_idx, window=win)
                        mask = alpha >= 250
                    elif nodata_val is not None:
                        mask = ~np.any(rgb == nodata_val, axis=-1)
                    else:
                        mask = np.ones((r_e - r_s, c_e - c_s), dtype=bool)

                    if np.any(mask):
                        rgb_u8 = np.clip(rgb, 0, 255).astype(np.uint8)

                        rgb_valid = rgb_u8[mask]
                        hsv_valid = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2HSV)[mask]
                        exg_valid = calculate_exg(rgb_u8)[mask]

                        feat_valid = np.hstack([rgb_valid, hsv_valid, exg_valid.reshape(-1, 1)])
                        n_valid = feat_valid.shape[0]
                        probas_valid = np.zeros((n_valid, len(rf_model.classes_)), dtype=np.float32)

                        for i in range(0, n_valid, batch_size):
                            j = min(i + batch_size, n_valid)
                            probas_valid[i:j] = rf_model.predict_proba(feat_valid[i:j])

                        for class_name, config in classification_config.items():
                            class_idx = class_indices[config["label_value"]]
                            prob_map_chunk = np.full((r_e - r_s, c_e - c_s), np.nan, dtype=np.float32)
                            prob_map_chunk[mask] = probas_valid[:, class_idx]
                            prob_maps[class_name][r_s:r_e, c_s:c_e] = prob_map_chunk

                            confidence_threshold = config.get("confidence")
                            output_las_suffix = config.get("output_las_suffix")
                            if confidence_threshold is not None and output_las_suffix is not None:
                                rows_chunk, cols_chunk = np.where((prob_map_chunk < confidence_threshold) & mask)
                                if len(rows_chunk) > 0:
                                    global_rows = rows_chunk + r_s
                                    global_cols = cols_chunk + c_s
                                    xs_pt, ys_pt = rasterio.transform.xy(src.transform, global_rows, global_cols)
                                    confidences = prob_map_chunk[rows_chunk, cols_chunk]
                                    all_confidence_points[class_name].append(np.vstack([xs_pt, ys_pt, confidences]).T)

                    pbar.update(1)

        class_time = time.time() - t_pipeline_start
        log_message(f"  -> Tempo de classificacao e coleta de pontos: {format_time(class_time)}", log_file)
        log_message(f"  -> Velocidade media: {(w*h)/class_time/1e6:.2f}M pixels/s", log_file)

        # Salvar TIFs
        las_paths = {}
        for class_name, config in classification_config.items():
            output_path = output_dir / f"{output_base_name}{config['output_tif_suffix']}"
            with rasterio.open(output_path, "w", **meta) as dst:
                dst.write(prob_maps[class_name], 1)
            log_message(f"     - {output_path} (P de ser {class_name})", log_file)

        # Salvar LAS
        for class_name, points_data_list in all_confidence_points.items():
            if points_data_list:
                all_points = np.vstack(points_data_list)
                header = laspy.LasHeader(point_format=3, version="1.2")
                header.add_extra_dims([laspy.ExtraBytesParams(name="confidence", type=np.float32)])
                min_x, min_y = np.min(all_points[:, 0]), np.min(all_points[:, 1])
                header.x_offset, header.y_offset, header.z_offset = min_x, min_y, 0.0
                header.x_scale, header.y_scale, header.z_scale = 0.001, 0.001, 0.001

                las = laspy.LasData(header)
                las.x, las.y = all_points[:, 0], all_points[:, 1]
                las.z = np.zeros_like(all_points[:, 2])
                las.confidence = all_points[:, 2]

                output_las_path = output_dir / f"{output_base_name}{classification_config[class_name]['output_las_suffix']}"
                try:
                    las.write(output_las_path)
                    log_message(f"  -> Pontos de confiança para {class_name} salvos em: {output_las_path}", log_file)
                    las_paths[class_name] = str(output_las_path)
                except laspy.errors.LaspyException as e:
                    if "No LazBackend selected" in str(e):
                        output_las_path = output_las_path.with_suffix(".las")
                        log_message(f"  -> Backend LAZ não encontrado. Salvando como LAS: {output_las_path}", log_file)
                        las.write(output_las_path)
                        las_paths[class_name] = str(output_las_path)
                    else:
                        raise e
            else:
                log_message(f"  -> Nenhum ponto de confiança encontrado para {class_name}.", log_file)

    t_pipeline = time.time() - t_pipeline_start
    log_message(f"\n  -> Tempo total de geracao: {format_time(t_pipeline)}", log_file)
    return las_paths


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

def generate_mdt_idw(xs, ys, zs, bounds, resolution, epsg, output_dir, base_name, log_file):
    log_message("\n" + "=" * 60, log_file)
    log_message("ETAPA 5: GERACAO DO MDT POR IDW + KDTREE", log_file)
    log_message("=" * 60, log_file)
    t0 = time.time()

    left, bottom, right, top = bounds
    width = int(np.ceil((right - left) / resolution))
    height = int(np.ceil((top - bottom) / resolution))

    log_message(f"   -> Resolucao do MDT: {resolution}m", log_file)
    log_message(f"   -> Bounds: left={left:.2f}, bottom={bottom:.2f}, right={right:.2f}, top={top:.2f}", log_file)
    log_message(f"   -> Dimensoes raster: {width} x {height} pixels", log_file)
    log_message(f"   -> Total de pixels: {width * height:,}", log_file)
    log_message(f"   -> Pontos de entrada: {len(xs):,}", log_file)
    log_message(f"   -> Vizinhos IDW: {IDW_K_NEIGHBORS}", log_file)
    log_message(f"   -> Potencia IDW: {IDW_POWER}", log_file)

    log_message("\n   Construindo KDTree...", log_file)
    t_kd = time.time()
    tree = KDTree(np.column_stack([xs, ys]))
    log_message(f"   -> KDTree construida em {format_time(time.time() - t_kd)}", log_file)

    transform = rasterio.transform.from_origin(left, top, resolution, resolution)
    meta = {
        "driver": "GTiff", "dtype": "float32", "nodata": np.nan,
        "width": width, "height": height, "count": 1, "crs": epsg,
        "transform": transform, "compress": "lzw", "tiled": True,
        "blockxsize": MDT_CHUNK_SIZE, "blockysize": MDT_CHUNK_SIZE,
    }

    output_path = os.path.join(output_dir, f"{base_name}.tif")
    log_message(f"\n   Processando IDW por chunks de {MDT_CHUNK_SIZE}x{MDT_CHUNK_SIZE}...", log_file)

    num_chunks_h = int(np.ceil(height / MDT_CHUNK_SIZE))
    num_chunks_w = int(np.ceil(width / MDT_CHUNK_SIZE))
    total_chunks = num_chunks_h * num_chunks_w

    with rasterio.open(output_path, "w", **meta) as dst:
        with tqdm(total=total_chunks, desc="IDW chunks", unit="chunk") as pbar:
            for r0 in range(0, height, MDT_CHUNK_SIZE):
                r1 = min(r0 + MDT_CHUNK_SIZE, height)
                for c0 in range(0, width, MDT_CHUNK_SIZE):
                    c1 = min(c0 + MDT_CHUNK_SIZE, width)

                    chunk_h = r1 - r0
                    chunk_w = c1 - c0

                    pixel_x = left + (c0 + np.arange(chunk_w) + 0.5) * resolution
                    pixel_y = top - (r0 + np.arange(chunk_h) + 0.5) * resolution
                    grid_x, grid_y = np.meshgrid(pixel_x, pixel_y)
                    query_points = np.column_stack([grid_x.ravel(), grid_y.ravel()])

                    distances, indices = tree.query(query_points, k=min(IDW_K_NEIGHBORS, len(xs)),
                                                     workers=-1, eps=0.0)

                    if distances.ndim == 1:
                        distances = distances[:, np.newaxis]
                        indices = indices[:, np.newaxis]

                    with np.errstate(divide='ignore', invalid='ignore'):
                        weights = 1.0 / (distances ** IDW_POWER + 1e-12)
                        weights[distances == 0] = 1e12

                        z_neighbors = zs[indices]
                        weighted_sum = np.nansum(z_neighbors * weights, axis=1)
                        weight_sum = np.nansum(weights, axis=1)

                        valid = weight_sum > 0
                        chunk_idw = np.full(chunk_h * chunk_w, np.nan, dtype=np.float32)
                        chunk_idw[valid] = weighted_sum[valid] / weight_sum[valid]

                        zero_dist = distances[:, 0:1].ravel() == 0
                        if np.any(zero_dist):
                            chunk_idw[zero_dist] = zs[indices[zero_dist, 0]]

                    dst.write(chunk_idw.reshape(chunk_h, chunk_w), 1,
                              window=rasterio.windows.Window(c0, r0, chunk_w, chunk_h))
                    pbar.update(1)

    t_total = time.time() - t0
    log_message(f"\n   -> MDT gerado em: {output_path}", log_file)
    log_message(f"   -> Tempo total IDW: {format_time(t_total)}", log_file)

    with rasterio.open(output_path) as result:
        data = result.read(1)
        valid_data = data[~np.isnan(data)]
        if len(valid_data) > 0:
            log_message(f"   -> MDT Z min: {np.nanmin(valid_data):.2f}m", log_file)
            log_message(f"   -> MDT Z max: {np.nanmax(valid_data):.2f}m", log_file)
            log_message(f"   -> MDT Z medio: {np.nanmean(valid_data):.2f}m", log_file)
            log_message(f"   -> MDT STD: {np.nanstd(valid_data):.2f}m", log_file)
            log_message(f"   -> Pixels preenchidos: {len(valid_data):,} de {data.size:,} ({100*len(valid_data)/data.size:.1f}%)", log_file)

    return output_path


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    start = time.time()
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    input_path = Path(INPUT_IMAGE_PATH)
    output_base_name = input_path.stem

    log_path = output_dir / f"{output_base_name}_process_info.txt"
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"\n{'#' * 80}\n# EXECUCAO COMPLETA (RF + MDT): {time.strftime('%d/%m/%Y %H:%M:%S')}\n{'#' * 80}\n\n")

    log_message(f"Arquivo de log: {log_path}", log_path)
    log_message(f"Imagem de entrada: {INPUT_IMAGE_PATH}", log_path)
    log_message(f"MDS de referencia: {MDS_PATH}", log_path)
    for class_name, config in CLASSIFICATION_CONFIG.items():
        if config.get("shp_path"):
            log_message(f"Pontos de treino de {class_name}: {config['shp_path']}", log_path)

    try:
        # --- ETAPA 1: Features ---
        features, labels = extract_features_for_training(INPUT_IMAGE_PATH, CLASSIFICATION_CONFIG, BUFFER_SIZE_METERS, log_path)
        if len(features) == 0:
            raise ValueError("Nenhuma feature extraida.")

        # --- ETAPA 2: Treinamento ---
        model = train_model(features, labels, log_path)

        # --- ETAPA 3: Classificacao + LAS ---
        las_paths = generate_probability_maps(INPUT_IMAGE_PATH, model, CLASSIFICATION_CONFIG, OUTPUT_DIR, log_path)

        # --- Determinar EPSG ---
        log_message("\n" + "=" * 60, log_path)
        log_message("ETAPA: DETERMINACAO DO EPSG PARA MDT", log_path)
        log_message("=" * 60, log_path)
        epsg = FORCE_EPSG
        # Tentar do LAS gerado se existir
        if "solo" in las_paths:
            epsg_las = get_epsg_from_las(las_paths["solo"])
            if epsg_las:
                epsg = epsg_las
        # Se nao, tentar do MDS
        if epsg is None:
            with rasterio.open(MDS_PATH) as mds_src:
                if mds_src.crs:
                    epsg = str(mds_src.crs)
        if epsg is None:
            epsg = FORCE_EPSG
        log_message(f"   -> EPSG utilizado: {epsg}", log_path)

        # --- ETAPA 4: Extrair Z do MDS ---
        if "solo" not in las_paths:
            raise ValueError("LAS de solo nao foi gerado. Nao e possivel criar MDT.")
        las_solo_path = las_paths["solo"]
        xs, ys, zs = extract_z_from_mds(las_solo_path, MDS_PATH, log_path)

        if len(zs) == 0:
            raise ValueError("Nenhum Z valido extraido do MDS.")

        # --- ETAPA 5: Gerar MDT ---
        with rasterio.open(MDS_PATH) as mds_src:
            bounds = mds_src.bounds

        mdt_base_name = f"{output_base_name}_MDT"
        mdt_path = generate_mdt_idw(xs, ys, zs, bounds, MDT_RESOLUTION, epsg, OUTPUT_DIR, mdt_base_name, log_path)

        total_time = time.time() - start
        log_message("\n" + "=" * 60, log_path)
        log_message(f"PROCESSO COMPLETO (RF + MDT) CONCLUIDO em {format_time(total_time)}", log_path)
        log_message("=" * 60, log_path)

    except Exception as e:
        log_message(f"\nERRO: {str(e)}", log_path)
        import traceback
        traceback.print_exc()
        raise