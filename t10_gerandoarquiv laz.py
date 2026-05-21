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

# =============================================================================
# CONFIGURACOES GLOBAIS
# =============================================================================
INPUT_IMAGE_PATH = r"D:\TESTES_PYTHON\OPEM_CV_TESTE\imaru\Imaru2.tif"
BUFFER_SIZE_METERS = 1
OUTPUT_DIR = r"D:\TESTES_PYTHON\OPEM_CV_TESTE\v11_final_"

# Novo dicionário de configuração para classes
CLASSIFICATION_CONFIG = {
    "floresta": {
        "shp_path": r"D:\TESTES_PYTHON\OPEM_CV_TESTE\imaru\floresta_pts.shp",
        "output_tif_suffix": "_prob_floresta.tif",
        "output_las_suffix": None, # Não gera pontos LAS para floresta
        "confidence": None, 
        "label_value": 1 # Valor numérico para a classe floresta
    },
    "solo": {
        "shp_path": r"D:\TESTES_PYTHON\OPEM_CV_TESTE\imaru\solo_pts.shp",
        "output_tif_suffix": "_prob_solo.tif",
        "output_las_suffix": "_solo_confidence.laz", # Gera pontos LAS para solo
        "confidence": 0.015, # Gera pontos para solo com confidence < 0.015
        "label_value": 0 # Valor numérico para a classe solo
    }
}

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
    """Excess Green Index: 2*G - R - B (normalized RGB)"""
    r, g, b = (
        img_rgb[:, :, 0].astype(float),
        img_rgb[:, :, 1].astype(float),
        img_rgb[:, :, 2].astype(float),
    )
    sum_rgb = r + g + b
    sum_rgb[sum_rgb == 0] = 1
    r_n, g_n, b_n = r / sum_rgb, g / sum_rgb, b / sum_rgb
    return 2 * g_n - r_n - b_n


def extract_features_for_training(
    tiff_path, classification_config, buffer_size_m, log_file
):
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
        log_message(
            f"Dimensoes da imagem: {w_img} x {h_img} pixels ({n_bands} bandas)",
            log_file,
        )
        log_message(f"Resolucao (GSD): {gsd:.6f} m/pixel", log_file)
        log_message(f"CRS: {crs_str}", log_file)
        log_message(
            f"Extent (bounds): left={bounds.left:.2f}, bottom={bounds.bottom:.2f}, right={bounds.right:.2f}, top={bounds.top:.2f}",
            log_file,
        )
        log_message(
            f"Buffer de treinamento: {buffer_size_m}m -> {buffer_px} pixels", log_file
        )
        log_message(f"Usando bandas 1(R), 2(G), 3(B) para features", log_file)

        img_rgb = np.moveaxis(src.read([1, 2, 3]), 0, -1)
        img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
        img_exg = calculate_exg(img_rgb)

        n_features = 7  # R, G, B, H, S, V, ExG

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
                            feat = np.hstack(
                                [
                                    rgb[r_s:r_e, c_s:c_e].reshape(-1, 3),
                                    hsv[r_s:r_e, c_s:c_e].reshape(-1, 3),
                                    exg[r_s:r_e, c_s:c_e].reshape(-1, 1),
                                ]
                            )
                            f_list.append(feat)
                            l_list.append(np.full(feat.shape[0], label_val))
                            valid_count += 1
                        else:
                            out_of_bounds += 1
                log_message(
                    f"  -> {name}: {valid_count} pontos dentro da imagem, {out_of_bounds} fora dos limites",
                    log_file,
                )
                return f_list, l_list
            else:
                log_message(f"  -> Arquivo de {name} nao encontrado: {path}", log_file)
                return [], []

        for class_name, config in classification_config.items():
            f_class, l_class = process_class(
                config["shp_path"], config["label_value"], class_name, src, img_rgb, img_hsv, img_exg, buffer_px
            )
            all_features.extend(f_class)
            all_labels.extend(l_class)

    if not all_features:
        log_message(
            "ERRO: Nenhuma feature extraida! Verifique os arquivos de pontos.", log_file
        )
        return np.empty((0, n_features)), np.empty((0,))

    total_pixels = sum(f.shape[0] for f in all_features)
    log_message(f"\nResumo da extracao:", log_file)
    log_message(
        f"  -> Features por pixel: R(1) G(1) B(1) | H(1) S(1) V(1) | ExG(1) = {n_features} features",
        log_file,
    )
    for class_name, config in classification_config.items():
        class_pixels = sum(np.sum(l == config["label_value"]) for l in all_labels if len(l) > 0)
        log_message(f"  -> Total de pixels de {class_name} para treino: {class_pixels:,}", log_file)

    log_message(f"  -> Total de amostras (pixels): {total_pixels:,}", log_file)

    return np.vstack(all_features), np.concatenate(all_labels)


def train_model(features, labels, log_file):
    log_message("\n" + "=" * 60, log_file)
    log_message("ETAPA 2: TREINAMENTO DO MODELO (Random Forest)", log_file)
    log_message("=" * 60, log_file)
    t0 = time.time()

    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, random_state=42
    )
    log_message(f"Tamanho do conjunto de treino: {X_train.shape[0]:,} pixels", log_file)
    log_message(f"Tamanho do conjunto de teste: {X_test.shape[0]:,} pixels", log_file)
    
    # Log distribution for each class
    unique_labels = np.unique(labels)
    for label_val in unique_labels:
        log_message(
            f"Distribuicao treino - Classe {label_val}: {np.sum(y_train==label_val)}",
            log_file,
        )
        log_message(
            f"Distribuicao teste  - Classe {label_val}: {np.sum(y_test==label_val)}",
            log_file,
        )

    log_message("Treinando Random Forest (n_estimators=100, max_depth=20)...", log_file)
    model = RandomForestClassifier(
        n_estimators=100, n_jobs=-1, max_depth=20, random_state=42
    )
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
    importances = sorted(
        zip(feature_names, model.feature_importances_), key=lambda x: -x[1]
    )
    for name, imp in importances:
        log_message(f"  {name}: {imp:.4f} ({imp*100:.1f}%)", log_file)

    log_message(
        f"\nClasses do modelo (ordem predict_proba): {model.classes_}", log_file
    )
    log_message(f"Numero de classes: {len(model.classes_)}", log_file)

    return model

def generate_probability_maps(
    tiff_path,
    rf_model,
    classification_config,
    out_dir,
    log_file,
    chunk_size=1024,
    batch_size=200000,
):
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
    log_message(f"  -> Features: R, G, B, H, S, V, ExG (7 features)", log_file)

    t_pipeline_start = time.time()

    # Mapear label_value para o índice de probabilidade
    class_indices = {config["label_value"]: np.where(rf_model.classes_ == config["label_value"])[0][0]
                     for class_name, config in classification_config.items()}

    log_message(f"\n[INFO] Mapeamento de classes: rf_model.classes_ = {rf_model.classes_}", log_file)
    for class_name, config in classification_config.items():
        log_message(
            f"[INFO] {class_name}_class_idx = {class_indices[config['label_value']]} (probabilidade de {class_name}, classe {config['label_value']})",
            log_file,
        )

    with rasterio.open(tiff_path) as src:
        h, w = src.shape
        meta = src.meta.copy()
        meta.update(dtype="float32", count=1)

        prob_maps = {class_name: np.zeros((h, w), dtype=np.float32)
                     for class_name in classification_config.keys()}
        
        # Preparar para coletar pontos de confiança
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
                    rgb_u8 = np.clip(rgb, 0, 255).astype(np.uint8)
                    feat = np.hstack(
                        [
                            rgb_u8.reshape(-1, 3),
                            cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2HSV).reshape(-1, 3),
                            calculate_exg(rgb_u8).reshape(-1, 1),
                        ]
                    )

                    n = feat.shape[0]
                    probas = np.zeros((n, len(rf_model.classes_)), dtype=np.float32)
                    for i in range(0, n, batch_size):
                        j = min(i + batch_size, n)
                        probas[i:j] = rf_model.predict_proba(feat[i:j])

                    for class_name, config in classification_config.items():
                        prob_map_chunk = probas[:, class_indices[config["label_value"]]].reshape(
                            r_e - r_s, c_e - c_s
                        )
                        prob_maps[class_name][r_s:r_e, c_s:c_e] = prob_map_chunk

                        # Extração de pontos de confiança para LAS/LAZ
                        confidence_threshold = config.get("confidence")
                        output_las_suffix = config.get("output_las_suffix")

                        if confidence_threshold is not None and output_las_suffix is not None:
                            # Encontrar pixels que atendem ao critério de confiança
                            # prob_map < confidence_threshold
                            rows_chunk, cols_chunk = np.where(prob_map_chunk < confidence_threshold)
                            
                            if len(rows_chunk) > 0:
                                # Converter coordenadas de pixel para geográficas
                                # Ajustar rows_chunk e cols_chunk para a posição global na imagem
                                global_rows = rows_chunk + r_s
                                global_cols = cols_chunk + c_s

                                # Vetorização da conversão de coordenadas
                                xs, ys = rasterio.transform.xy(src.transform, global_rows, global_cols)
                                
                                # Coletar as confianças correspondentes
                                confidences = prob_map_chunk[rows_chunk, cols_chunk]

                                # Armazenar os pontos e confianças
                                all_confidence_points[class_name].append(
                                    np.vstack([xs, ys, confidences]).T
                                )
                    pbar.update(1)

        class_time = time.time() - t_pipeline_start # Tempo total para classificação e coleta de pontos
        log_message(f"  -> Tempo de classificacao e coleta de pontos: {format_time(class_time)}", log_file)
        log_message(
            f"  -> Velocidade media: {(w*h)/class_time/1e6:.2f}M pixels/s", log_file
        )

        output_paths = {}
        # Salvar mapas de probabilidade TIF
        for class_name, config in classification_config.items():
            output_path = output_dir / f"{output_base_name}{config['output_tif_suffix']}"
            with rasterio.open(output_path, "w", **meta) as dst:
                dst.write(prob_maps[class_name], 1)
            output_paths[class_name] = output_path
            log_message(f"     - {output_path} (P de ser {class_name})", log_file)

        # Salvar pontos de confiança LAS/LAZ
        for class_name, points_data_list in all_confidence_points.items():
            if points_data_list:
                all_points_for_class = np.vstack(points_data_list)
                
                # Criar um cabeçalho LAS
                header = laspy.LasHeader(point_format=3, version="1.2") # Formato 3 inclui intensidade, RGB, tempo GPS
                header.add_extra_dims([laspy.ExtraBytesParams(name="confidence", type=np.float32)])
                
                # Definir offsets e escalas para coordenadas
                min_x, min_y = np.min(all_points_for_class[:, 0]), np.min(all_points_for_class[:, 1])
                
                header.x_offset = min_x
                header.y_offset = min_y
                header.z_offset = 0.0
                header.x_scale = 0.001
                header.y_scale = 0.001
                header.z_scale = 0.001

                las = laspy.LasData(header)
                las.x = all_points_for_class[:, 0]
                las.y = all_points_for_class[:, 1]
                las.z = np.zeros_like(all_points_for_class[:, 2])
                las.confidence = all_points_for_class[:, 2]

                output_las_path = output_dir / f"{output_base_name}{classification_config[class_name]['output_las_suffix']}"
                
                # Tentar salvar como LAZ, se falhar por falta de backend, salvar como LAS
                try:
                    las.write(output_las_path)
                    log_message(f"  -> Pontos de confiança para {class_name} salvos em: {output_las_path}", log_file)
                except laspy.errors.LaspyException as e:
                    if "No LazBackend selected" in str(e):
                        # Mudar extensão para .las
                        output_las_path = output_las_path.with_suffix(".las")
                        log_message(f"  -> Backend LAZ não encontrado. Salvando como LAS sem compressão: {output_las_path}", log_file)
                        las.write(output_las_path)
                    else:
                        raise e
            else:
                log_message(f"  -> Nenhum ponto de confiança encontrado para {class_name} para salvar em LAS/LAZ.", log_file)

    t_pipeline = time.time() - t_pipeline_start
    log_message(
        f"\n  -> Tempo total de geracao de mapas e pontos: {format_time(t_pipeline)}", log_file
    )

    return output_paths


if __name__ == "__main__":
    start = time.time()

    # Define o arquivo de log no diretorio de saida
    log_dir = Path(OUTPUT_DIR)
    log_file_path = log_dir / f"{Path(INPUT_IMAGE_PATH).stem}_process_info.txt"
    os.makedirs(log_dir, exist_ok=True)

    with open(log_file_path, "a", encoding="utf-8") as f:
        f.write(f"\n{'#' * 80}\n")
        f.write(f"# NOVA EXECUCAO: {time.strftime('%d/%m/%Y %H:%M:%S')}\n")
        f.write(f"# SCRIPT: generate_probability_maps_and_points.py\n")
        f.write(f"{'#' * 80}\n\n")

    log_message(f"Arquivo de log: {log_file_path}", log_file_path)
    log_message(f"Imagem de entrada: {INPUT_IMAGE_PATH}", log_file_path)
    
    for class_name, config in CLASSIFICATION_CONFIG.items():
        if config.get("shp_path"):
            log_message(f"Pontos de treino de {class_name}: {config['shp_path']}", log_file_path)

    log_message(f"Features: R, G, B, H, S, V, ExG (7 features)", log_file_path)

    try:
        features, labels = extract_features_for_training(
            INPUT_IMAGE_PATH,
            CLASSIFICATION_CONFIG,
            BUFFER_SIZE_METERS,
            log_file_path,
        )
        if len(features) > 0:
            model = train_model(features, labels, log_file_path)
            generate_probability_maps(
                INPUT_IMAGE_PATH,
                model,
                CLASSIFICATION_CONFIG,
                OUTPUT_DIR,
                log_file_path,
            )
        else:
            log_message(
                "Nao foi possivel extrair features para treinamento. Encerrando.",
                log_file_path,
            )

        total_time = time.time() - start
        log_message("\n" + "=" * 60, log_file_path)
        log_message(f"PROCESSO CONCLUIDO em {format_time(total_time)}", log_file_path)
        log_message(f"Termino: {time.strftime('%d/%m/%Y %H:%M:%S')}", log_file_path)
        log_message("=" * 60, log_file_path)
    except Exception as e:
        log_message(f"\nERRO: {str(e)}", log_file_path)
        log_message(
            "Ocorreu um erro. Verifique o log para mais detalhes.", log_file_path
        )
        raise
