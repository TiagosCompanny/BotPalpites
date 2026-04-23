import os
import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans


BASE_DIR = Path(__file__).resolve().parent
ARQUIVO_BASE = BASE_DIR / "DadosRodovias" / "dados_todas_rodovias.xlsx"
MODELOS_DIR = BASE_DIR / "modelos_rodovias"

CATALOGO_RODOVIAS = {
    "braganca_paulista": {
        "rodovia_identificacao": "Rodovia Arão Sahm, KM 95 — Bragança Paulista (SP).",
        "pasta_saida": str(MODELOS_DIR / "rodovia_arao_sahm_km_95_braganca_paulista_sp_rf_regimes"),
        "filtro_rodovia_legado": "Rodovia Arão Sahm",
        "nome_exibicao": "BRAGANÇA PAULISTA"
    },
    "caraguatatuba": {
        "rodovia_identificacao": "Doutor Manoel Hyppolito Rego, KM 83 — Caraguatatuba (SP).",
        "pasta_saida": str(MODELOS_DIR / "doutor_manoel_hyppolito_rego_km_83_caraguatatuba_sp_rf_regimes"),
        "filtro_rodovia_legado": "Doutor Manoel Hyppolito Rego",
        "nome_exibicao": "CARAGUATATUBA"
    },
    "pindamonhangaba": {
        "rodovia_identificacao": "Floriano Rodrigues Pinheiro, KM 26 — Pindamonhangaba (SP).",
        "pasta_saida": str(MODELOS_DIR / "floriano_rodrigues_pinheiro_km_26_pindamonhangaba_sp_rf_regimes"),
        "filtro_rodovia_legado": "Floriano Rodrigues Pinheiro",
        "nome_exibicao": "PINDAMONHANGABA"
    }
}


def parse_float(x):
    try:
        return float(str(x).replace(",", "."))
    except Exception:
        return np.nan


def obter_config_rodovia(rodovia_chave):
    if rodovia_chave not in CATALOGO_RODOVIAS:
        chaves_disponiveis = ", ".join(CATALOGO_RODOVIAS.keys())
        raise ValueError(f"Rodovia '{rodovia_chave}' não encontrada no catálogo. Disponíveis: {chaves_disponiveis}")
    return CATALOGO_RODOVIAS[rodovia_chave]


def preparar_base(df, config):
    df = df.copy()

    for col in ["abertura", "fechamento", "resolvido_em"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    df = df.dropna(subset=["rodovia_identificacao", "abertura", "resultado_vencedor"]).copy()

    rodovia_identificacao = config["rodovia_identificacao"]
    filtro_rodovia_legado = config.get("filtro_rodovia_legado")

    if "rodovia_identificacao" in df.columns:
        df = df[df["rodovia_identificacao"] == rodovia_identificacao].copy()
    elif "rodovia" in df.columns and filtro_rodovia_legado:
        df = df[df["rodovia"].astype(str).str.contains(filtro_rodovia_legado, case=False, na=False)].copy()

    if df.empty and "rodovia" in df.columns and filtro_rodovia_legado:
        df = df[df["rodovia"].astype(str).str.contains(filtro_rodovia_legado, case=False, na=False)].copy()

    if df.empty:
        raise ValueError(f"Nenhum registro encontrado para a rodovia: {rodovia_identificacao}")

    df = df.sort_values("abertura").reset_index(drop=True)

    df["alvo_mais"] = df["resultado_vencedor"].astype(str).str.contains("Mais", case=False, na=False).astype(int)

    colunas_numericas = [
        "meta_referencia",
        "fim_semana",
        "hora_sin",
        "hora_cos",
        "dia_semana_sin",
        "dia_semana_cos",
        "mes_sin",
        "mes_cos",
        "temperatura_2m",
        "umidade_relativa",
        "chuva_mm",
        "cobertura_nuvens",
        "estava_chovendo"
    ]

    for col in colunas_numericas:
        if col in df.columns:
            df[col] = df[col].apply(parse_float)

    if "estava_chovendo" in df.columns:
        df["estava_chovendo"] = df["estava_chovendo"].fillna(0)

    df["meta_num"] = df["meta_referencia"].apply(parse_float)

    df["hora"] = df["abertura"].dt.hour
    df["minuto"] = df["abertura"].dt.minute
    df["dia_semana"] = df["abertura"].dt.dayofweek
    df["mes"] = df["abertura"].dt.month
    df["hora_decimal"] = df["hora"] + (df["minuto"] / 60.0)

    if "fim_semana" not in df.columns:
        df["fim_semana"] = df["dia_semana"].isin([5, 6]).astype(int)

    if "hora_sin" not in df.columns:
        df["hora_sin"] = np.sin(2 * np.pi * df["hora_decimal"] / 24)
        df["hora_cos"] = np.cos(2 * np.pi * df["hora_decimal"] / 24)

    if "dia_semana_sin" not in df.columns:
        df["dia_semana_sin"] = np.sin(2 * np.pi * df["dia_semana"] / 7)
        df["dia_semana_cos"] = np.cos(2 * np.pi * df["dia_semana"] / 7)

    if "mes_sin" not in df.columns:
        df["mes_sin"] = np.sin(2 * np.pi * (df["mes"] - 1) / 12)
        df["mes_cos"] = np.cos(2 * np.pi * (df["mes"] - 1) / 12)

    return df


def criar_bloco_dia(df):
    hora = df["hora_decimal"]

    condicoes = [
        (hora >= 0) & (hora < 5),
        (hora >= 5) & (hora < 8),
        (hora >= 8) & (hora < 11),
        (hora >= 11) & (hora < 14),
        (hora >= 14) & (hora < 18),
        (hora >= 18) & (hora < 22),
        (hora >= 22) & (hora <= 23.999999)
    ]

    valores = [
        "madrugada",
        "pico_manha",
        "manha",
        "almoco",
        "tarde_pico",
        "noite",
        "late_night"
    ]

    df["bloco_dia"] = np.select(condicoes, valores, default="outro")
    df["bloco_dia_cod"] = pd.Categorical(
        df["bloco_dia"],
        categories=["madrugada", "pico_manha", "manha", "almoco", "tarde_pico", "noite", "late_night", "outro"]
    ).codes

    dummies = pd.get_dummies(df["bloco_dia"], prefix="bloco")
    df = pd.concat([df, dummies], axis=1)

    return df


def criar_features(df):
    df = df.copy()
    df = criar_bloco_dia(df)

    df["eh_pico_manha"] = df["hora"].between(5, 8).astype(int)
    df["eh_pico_tarde"] = df["hora"].between(16, 18).astype(int)
    df["eh_sexta"] = (df["dia_semana"] == 4).astype(int)
    df["eh_domingo"] = (df["dia_semana"] == 6).astype(int)
    df["eh_sabado"] = (df["dia_semana"] == 5).astype(int)
    df["eh_dia_util"] = (df["dia_semana"] <= 4).astype(int)

    for lag in [1, 2, 3, 4, 6, 12]:
        df[f"meta_lag_{lag}"] = df["meta_num"].shift(lag)
        df[f"alvo_lag_{lag}"] = df["alvo_mais"].shift(lag)

    df["meta_media_3"] = df["meta_num"].shift(1).rolling(3).mean()
    df["meta_media_4"] = df["meta_num"].shift(1).rolling(4).mean()
    df["meta_media_6"] = df["meta_num"].shift(1).rolling(6).mean()
    df["meta_media_12"] = df["meta_num"].shift(1).rolling(12).mean()

    df["meta_std_3"] = df["meta_num"].shift(1).rolling(3).std()
    df["meta_std_4"] = df["meta_num"].shift(1).rolling(4).std()
    df["meta_std_6"] = df["meta_num"].shift(1).rolling(6).std()
    df["meta_std_12"] = df["meta_num"].shift(1).rolling(12).std()

    df["meta_max_4"] = df["meta_num"].shift(1).rolling(4).max()
    df["meta_min_4"] = df["meta_num"].shift(1).rolling(4).min()
    df["meta_max_12"] = df["meta_num"].shift(1).rolling(12).max()
    df["meta_min_12"] = df["meta_num"].shift(1).rolling(12).min()

    df["meta_delta_1"] = df["meta_num"].shift(1) - df["meta_num"].shift(2)
    df["meta_delta_2"] = df["meta_num"].shift(1) - df["meta_num"].shift(3)
    df["meta_delta_4"] = df["meta_num"].shift(1) - df["meta_num"].shift(5)
    df["meta_delta_media_4"] = df["meta_num"].shift(1) - df["meta_media_4"]
    df["meta_delta_media_12"] = df["meta_num"].shift(1) - df["meta_media_12"]

    df["tendencia_curta_meta"] = df["meta_media_3"] - df["meta_media_6"]
    df["tendencia_media_meta"] = df["meta_media_4"] - df["meta_media_12"]
    df["acelerando_meta"] = (df["tendencia_curta_meta"] > 0).astype(float)

    df["media_alvo_3"] = df["alvo_mais"].shift(1).rolling(3).mean()
    df["media_alvo_6"] = df["alvo_mais"].shift(1).rolling(6).mean()
    df["media_alvo_12"] = df["alvo_mais"].shift(1).rolling(12).mean()

    df["soma_alvo_3"] = df["alvo_mais"].shift(1).rolling(3).sum()
    df["soma_alvo_6"] = df["alvo_mais"].shift(1).rolling(6).sum()
    df["soma_alvo_12"] = df["alvo_mais"].shift(1).rolling(12).sum()

    df["tendencia_curta_1_2"] = df["alvo_mais"].shift(1) - df["alvo_mais"].shift(2)
    df["tendencia_curta_1_3"] = df["alvo_mais"].shift(1) - df["alvo_mais"].shift(3)
    df["tendencia_curta_1_6"] = df["alvo_mais"].shift(1) - df["alvo_mais"].shift(6)

    df["interacao_meta_hora_sin"] = df["meta_num"] * df["hora_sin"]
    df["interacao_meta_hora_cos"] = df["meta_num"] * df["hora_cos"]

    if "temperatura_2m" in df.columns:
        df["interacao_meta_temperatura"] = df["meta_num"] * df["temperatura_2m"]
    else:
        df["interacao_meta_temperatura"] = np.nan

    if "umidade_relativa" in df.columns:
        df["interacao_meta_umidade"] = df["meta_num"] * df["umidade_relativa"]
    else:
        df["interacao_meta_umidade"] = np.nan

    return df


def calcular_metricas(y_true, prob):
    pred = (prob >= 0.5).astype(int)
    acc = accuracy_score(y_true, pred)
    baseline = max(y_true.mean(), 1 - y_true.mean())

    try:
        auc = roc_auc_score(y_true, prob)
    except Exception:
        auc = None

    return {
        "accuracy": round(float(acc), 6),
        "auc": round(float(auc), 6) if auc is not None else None,
        "precision_mais": round(float(precision_score(y_true, pred, zero_division=0)), 6),
        "recall_mais": round(float(recall_score(y_true, pred, zero_division=0)), 6),
        "f1_mais": round(float(f1_score(y_true, pred, zero_division=0)), 6),
        "f1_ate": round(float(f1_score(y_true, pred, pos_label=0, zero_division=0)), 6),
        "macro_f1": round(float(f1_score(y_true, pred, average="macro", zero_division=0)), 6),
        "baseline_majoritaria": round(float(baseline), 6),
        "ganho_vs_baseline": round(float(acc - baseline), 6),
        "taxa_mais_teste": round(float(y_true.mean()), 6)
    }


def adicionar_cluster_feature(treino, teste, colunas_cluster, n_clusters=4):
    treino = treino.copy()
    teste = teste.copy()

    imp_cluster = SimpleImputer(strategy="median")
    scaler_cluster = StandardScaler()

    X_cluster_treino = imp_cluster.fit_transform(treino[colunas_cluster])
    X_cluster_teste = imp_cluster.transform(teste[colunas_cluster])

    X_cluster_treino = scaler_cluster.fit_transform(X_cluster_treino)
    X_cluster_teste = scaler_cluster.transform(X_cluster_teste)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=20)
    treino["cluster_regime"] = kmeans.fit_predict(X_cluster_treino)
    teste["cluster_regime"] = kmeans.predict(X_cluster_teste)

    for c in range(n_clusters):
        treino[f"cluster_regime_{c}"] = (treino["cluster_regime"] == c).astype(int)
        teste[f"cluster_regime_{c}"] = (teste["cluster_regime"] == c).astype(int)

    return treino, teste, imp_cluster, scaler_cluster, kmeans


def treinar_modelo_rodovia(rodovia_chave, caminho_excel=ARQUIVO_BASE):
    config = obter_config_rodovia(rodovia_chave)

    rodovia_identificacao = config["rodovia_identificacao"]
    pasta_saida = config["pasta_saida"]
    nome_exibicao = config.get("nome_exibicao", rodovia_identificacao)

    os.makedirs(pasta_saida, exist_ok=True)

    df = pd.read_excel(caminho_excel)
    df = preparar_base(df, config)
    df = criar_features(df)

    colunas_base = [
        "meta_num",
        "fim_semana",
        "hora",
        "minuto",
        "hora_decimal",
        "dia_semana",
        "mes",
        "hora_sin",
        "hora_cos",
        "dia_semana_sin",
        "dia_semana_cos",
        "mes_sin",
        "mes_cos",
        "temperatura_2m",
        "umidade_relativa",
        "chuva_mm",
        "cobertura_nuvens",
        "estava_chovendo",
        "bloco_dia_cod",
        "eh_pico_manha",
        "eh_pico_tarde",
        "eh_sexta",
        "eh_domingo",
        "eh_sabado",
        "eh_dia_util",
        "meta_lag_1",
        "meta_lag_2",
        "meta_lag_3",
        "meta_lag_4",
        "meta_lag_6",
        "meta_lag_12",
        "meta_media_3",
        "meta_media_4",
        "meta_media_6",
        "meta_media_12",
        "meta_std_3",
        "meta_std_4",
        "meta_std_6",
        "meta_std_12",
        "meta_max_4",
        "meta_min_4",
        "meta_max_12",
        "meta_min_12",
        "meta_delta_1",
        "meta_delta_2",
        "meta_delta_4",
        "meta_delta_media_4",
        "meta_delta_media_12",
        "tendencia_curta_meta",
        "tendencia_media_meta",
        "acelerando_meta",
        "alvo_lag_1",
        "alvo_lag_2",
        "alvo_lag_3",
        "alvo_lag_4",
        "alvo_lag_6",
        "alvo_lag_12",
        "media_alvo_3",
        "media_alvo_6",
        "media_alvo_12",
        "soma_alvo_3",
        "soma_alvo_6",
        "soma_alvo_12",
        "tendencia_curta_1_2",
        "tendencia_curta_1_3",
        "tendencia_curta_1_6",
        "interacao_meta_hora_sin",
        "interacao_meta_hora_cos",
        "interacao_meta_temperatura",
        "interacao_meta_umidade"
    ]

    colunas_bloco = [
        c for c in df.columns
        if c.startswith("bloco_") and c != "bloco_dia"
    ]

    colunas_base = list(dict.fromkeys(colunas_base + colunas_bloco))
    colunas_base = [c for c in colunas_base if c in df.columns]
    colunas_base = [c for c in colunas_base if pd.api.types.is_numeric_dtype(df[c])]

    df_model = df.dropna(subset=["alvo_mais"]).copy()

    colunas_minimas = [
        "meta_num",
        "hora_sin",
        "hora_cos",
        "dia_semana_sin",
        "dia_semana_cos",
        "meta_lag_1",
        "meta_media_4",
        "meta_media_12",
        "alvo_lag_1"
    ]
    colunas_minimas = [c for c in colunas_minimas if c in df_model.columns]
    df_model = df_model.dropna(subset=colunas_minimas).copy()

    n = len(df_model)
    if n < 60:
        raise ValueError(f"Poucos registros para modelagem: {n}")

    split = int(n * 0.8)
    treino = df_model.iloc[:split].copy()
    teste = df_model.iloc[split:].copy()

    if treino["alvo_mais"].nunique() < 2:
        raise ValueError("Treino sem duas classes.")

    if teste["alvo_mais"].nunique() < 2:
        raise ValueError("Teste sem duas classes.")

    colunas_cluster = [
        "meta_num",
        "hora_sin",
        "hora_cos",
        "dia_semana_sin",
        "dia_semana_cos",
        "fim_semana",
        "meta_media_4",
        "meta_media_12",
        "meta_std_4",
        "meta_std_12",
        "meta_delta_1",
        "meta_delta_4",
        "eh_pico_manha",
        "eh_pico_tarde"
    ]
    colunas_cluster = [c for c in colunas_cluster if c in treino.columns]

    treino, teste, imp_cluster, scaler_cluster, kmeans = adicionar_cluster_feature(
        treino=treino,
        teste=teste,
        colunas_cluster=colunas_cluster,
        n_clusters=4
    )

    colunas_cluster_rf = ["cluster_regime"] + [f"cluster_regime_{c}" for c in range(4)]
    features = list(dict.fromkeys(colunas_base + colunas_cluster_rf))
    features = [c for c in features if c in treino.columns]

    X_treino = treino[features]
    y_treino = treino["alvo_mais"]

    X_teste = teste[features]
    y_teste = teste["alvo_mais"]

    modelo = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("model", RandomForestClassifier(
            n_estimators=1200,
            max_depth=20,
            min_samples_split=2,
            min_samples_leaf=1,
            max_features="sqrt",
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=-1
        ))
    ])

    modelo.fit(X_treino, y_treino)
    prob = modelo.predict_proba(X_teste)[:, 1]
    pred = (prob >= 0.5).astype(int)

    metricas = calcular_metricas(y_teste, prob)

    print(f"\n=== RESULTADO FINAL - {nome_exibicao} ===")
    print(f"Base total: {n}")
    print(f"Base treino: {len(treino)}")
    print(f"Base teste: {len(teste)}")
    print(f"Acurácia: {metricas['accuracy'] * 100:.2f}%")
    print(f"Macro F1: {metricas['macro_f1']:.4f}")
    print(f"F1 Mais: {metricas['f1_mais']:.4f}")
    print(f"F1 Até: {metricas['f1_ate']:.4f}")
    print("\nRelatório de Classificação:")
    print(classification_report(y_teste, pred, target_names=["Até a Meta (0)", "Acima da Meta (1)"]))
    print("Matriz de Confusão:")
    print(confusion_matrix(y_teste, pred))

    modelo_final = modelo.named_steps["model"]
    importancias = pd.DataFrame({
        "variavel": features,
        "importancia": modelo_final.feature_importances_
    }).sort_values(by="importancia", ascending=False)

    print("\n=== IMPORTÂNCIA DAS VARIÁVEIS ===")
    print(importancias.head(30))

    caminho_modelo = os.path.join(pasta_saida, "modelo.joblib")
    caminho_metadata = os.path.join(pasta_saida, "metadata.json")
    caminho_predicoes = os.path.join(pasta_saida, "predicoes_teste.xlsx")
    caminho_importancias = os.path.join(pasta_saida, "importancias.xlsx")

    bundle = {
        "modelo": modelo,
        "features": features,
        "colunas_cluster": colunas_cluster,
        "imp_cluster": imp_cluster,
        "scaler_cluster": scaler_cluster,
        "kmeans": kmeans,
        "rodovia_identificacao": rodovia_identificacao,
        "rodovia_chave": rodovia_chave
    }

    joblib.dump(bundle, caminho_modelo)

    metadata = {
        "rodovia_chave": rodovia_chave,
        "rodovia_identificacao": rodovia_identificacao,
        "features": features,
        "target": "alvo_mais",
        "classe_1": "Mais de X",
        "classe_0": "Até X",
        "modelo_escolhido": "rf_regimes_clusters_lags",
        "split": "temporal_80_20",
        "metricas_teste": metricas,
        "quantidades": {
            "total_modelagem": int(n),
            "treino": int(len(treino)),
            "teste": int(len(teste))
        }
    }

    with open(caminho_metadata, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=4)

    predicoes = teste.copy()
    predicoes["prob_mais_modelo"] = prob
    predicoes["pred_mais"] = pred
    predicoes["classe_real"] = np.where(predicoes["alvo_mais"] == 1, "Mais de X", "Até X")
    predicoes["classe_prevista"] = np.where(predicoes["pred_mais"] == 1, "Mais de X", "Até X")
    predicoes["acertou"] = (predicoes["alvo_mais"] == predicoes["pred_mais"]).astype(int)

    predicoes.to_excel(caminho_predicoes, index=False)
    importancias.to_excel(caminho_importancias, index=False)

    print(f"\nModelo salvo em: {caminho_modelo}")
    print(f"Metadata salva em: {caminho_metadata}")
    print(f"Predições salvas em: {caminho_predicoes}")
    print(f"Importâncias salvas em: {caminho_importancias}")


if __name__ == "__main__":


    for rodovia_chave in CATALOGO_RODOVIAS.keys():
        print(f"\n\n==============================")
        print(f"INICIANDO MODELAGEM PARA: {rodovia_chave}")
        print(f"==============================\n\n")

        try:
            treinar_modelo_rodovia(
                rodovia_chave=rodovia_chave,
                caminho_excel=ARQUIVO_BASE
            )
        except Exception as e:
            print(f"Erro ao modelar rodovia '{rodovia_chave}': {e}")

    #braganca_paulista
    #caraguatatuba
    #pindamonhangaba
    """
    RODOVIA_CHAVE = "braganca_paulista"
    treinar_modelo_rodovia(
        rodovia_chave=RODOVIA_CHAVE,
        caminho_excel=ARQUIVO_BASE
    )
    
    """
    