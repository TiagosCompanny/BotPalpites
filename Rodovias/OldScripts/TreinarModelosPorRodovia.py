import os
import re
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


def normalizar_nome_arquivo(texto):
    texto = str(texto).strip().lower()
    texto = re.sub(r"[^\w\s-]", "", texto, flags=re.UNICODE)
    texto = re.sub(r"[\s]+", "_", texto)
    return texto


def parse_float(x):
    try:
        return float(str(x).replace(",", "."))
    except Exception:
        return np.nan


def preparar_base(df):
    df = df.copy()

    for col in ["abertura", "fechamento", "resolvido_em"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    df = df.dropna(subset=["rodovia_identificacao", "abertura", "resultado_vencedor"]).copy()

    df["alvo_mais"] = df["resultado_vencedor"].astype(str).str.startswith("Mais de").astype(int)

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

    if "meta_referencia" in df.columns:
        df["meta_num"] = df["meta_referencia"].apply(parse_float)
    else:
        df["meta_num"] = np.nan

    if "fim_semana" not in df.columns:
        df["fim_semana"] = df["abertura"].dt.dayofweek.isin([5, 6]).astype(int)

    if "hora_sin" not in df.columns:
        hora = df["abertura"].dt.hour
        df["hora_sin"] = np.sin(2 * np.pi * hora / 24)
        df["hora_cos"] = np.cos(2 * np.pi * hora / 24)

    if "dia_semana_sin" not in df.columns:
        dow = df["abertura"].dt.dayofweek
        df["dia_semana_sin"] = np.sin(2 * np.pi * dow / 7)
        df["dia_semana_cos"] = np.cos(2 * np.pi * dow / 7)

    if "mes_sin" not in df.columns:
        mes = df["abertura"].dt.month
        df["mes_sin"] = np.sin(2 * np.pi * (mes - 1) / 12)
        df["mes_cos"] = np.cos(2 * np.pi * (mes - 1) / 12)

    return df


def criar_features_grupo(g):
    g = g.sort_values("abertura").reset_index(drop=True).copy()

    g["alvo_lag_1"] = g["alvo_mais"].shift(1)
    g["alvo_lag_2"] = g["alvo_mais"].shift(2)
    g["alvo_lag_3"] = g["alvo_mais"].shift(3)
    g["alvo_lag_6"] = g["alvo_mais"].shift(6)
    g["alvo_lag_12"] = g["alvo_mais"].shift(12)

    g["media_alvo_3"] = g["alvo_mais"].shift(1).rolling(3).mean()
    g["media_alvo_6"] = g["alvo_mais"].shift(1).rolling(6).mean()
    g["media_alvo_12"] = g["alvo_mais"].shift(1).rolling(12).mean()

    g["soma_alvo_3"] = g["alvo_mais"].shift(1).rolling(3).sum()
    g["soma_alvo_6"] = g["alvo_mais"].shift(1).rolling(6).sum()
    g["soma_alvo_12"] = g["alvo_mais"].shift(1).rolling(12).sum()

    g["tendencia_curta_1_2"] = g["alvo_mais"].shift(1) - g["alvo_mais"].shift(2)
    g["tendencia_curta_1_3"] = g["alvo_mais"].shift(1) - g["alvo_mais"].shift(3)
    g["tendencia_curta_1_6"] = g["alvo_mais"].shift(1) - g["alvo_mais"].shift(6)

    g["interacao_meta_hora_sin"] = g["meta_num"] * g["hora_sin"]
    g["interacao_meta_hora_cos"] = g["meta_num"] * g["hora_cos"]

    if "temperatura_2m" in g.columns:
        g["interacao_meta_temperatura"] = g["meta_num"] * g["temperatura_2m"]
    else:
        g["interacao_meta_temperatura"] = np.nan

    if "umidade_relativa" in g.columns:
        g["interacao_meta_umidade"] = g["meta_num"] * g["umidade_relativa"]
    else:
        g["interacao_meta_umidade"] = np.nan

    return g


def criar_modelos():
    modelos = {
        "rf_rico": Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("model", RandomForestClassifier(
                n_estimators=400,
                max_depth=12,
                min_samples_split=10,
                min_samples_leaf=4,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1
            ))
        ]),
        "extra_trees_rico": Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("model", ExtraTreesClassifier(
                n_estimators=500,
                max_depth=12,
                min_samples_leaf=3,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1
            ))
        ]),
        "hist_gb_rico": Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("model", HistGradientBoostingClassifier(
                max_iter=300,
                learning_rate=0.05,
                max_depth=6,
                min_samples_leaf=20,
                random_state=42
            ))
        ]),
        "log_reg_rico": Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                random_state=42
            ))
        ])
    }

    return modelos


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


def score_modelo(metricas):
    return (
        metricas["macro_f1"],
        metricas["accuracy"],
        metricas["ganho_vs_baseline"],
        metricas["auc"] if metricas["auc"] is not None else -999
    )


def treinar_modelos_por_rodovia_escolhendo_melhor(
    caminho_excel,
    pasta_saida="modelos_rodovias_best",
    min_registros_por_rodovia=60
):
    os.makedirs(pasta_saida, exist_ok=True)

    df = pd.read_excel(caminho_excel)
    df = preparar_base(df)

    resumo_final = []
    comparativo_modelos = []
    modelos_base = criar_modelos()

    for rodovia, g in df.groupby("rodovia_identificacao"):
        g = criar_features_grupo(g)
        g = g.sort_values("abertura").reset_index(drop=True).copy()
        n_total_bruto = len(g)

        features = [
            "meta_num",
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
            "estava_chovendo",
            "alvo_lag_1",
            "alvo_lag_2",
            "alvo_lag_3",
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

        features = [col for col in features if col in g.columns]

        g = g.dropna(subset=["alvo_mais"]).copy()
        g = g.dropna(subset=features).copy()
        n = len(g)

        if n < min_registros_por_rodovia:
            print(f"[IGNORADO] {rodovia} | poucos registros após features: {n} de {n_total_bruto}")
            continue

        split = int(n * 0.8)

        if split <= 0 or split >= n:
            print(f"[IGNORADO] {rodovia} | split inválido")
            continue

        treino = g.iloc[:split].copy()
        teste = g.iloc[split:].copy()

        if treino["alvo_mais"].nunique() < 2:
            print(f"[IGNORADO] {rodovia} | treino sem duas classes")
            continue

        if teste["alvo_mais"].nunique() < 2:
            print(f"[IGNORADO] {rodovia} | teste sem duas classes")
            continue

        X_treino = treino[features]
        y_treino = treino["alvo_mais"]

        X_teste = teste[features]
        y_teste = teste["alvo_mais"]

        melhor = None

        for nome_modelo, modelo in modelos_base.items():
            modelo.fit(X_treino, y_treino)

            if hasattr(modelo, "predict_proba"):
                prob = modelo.predict_proba(X_teste)[:, 1]
            else:
                score = modelo.decision_function(X_teste)
                prob = 1 / (1 + np.exp(-score))

            metricas = calcular_metricas(y_teste, prob)

            comparativo_modelos.append({
                "rodovia_identificacao": rodovia,
                "modelo_testado": nome_modelo,
                "qtd_total_bruto": n_total_bruto,
                "qtd_total_modelagem": n,
                "qtd_treino": len(treino),
                "qtd_teste": len(teste),
                **metricas
            })

            atual = {
                "nome_modelo": nome_modelo,
                "pipeline": modelo,
                "metricas": metricas,
                "prob": prob
            }

            if melhor is None or score_modelo(metricas) > score_modelo(melhor["metricas"]):
                melhor = atual

        nome_pasta = normalizar_nome_arquivo(rodovia)
        pasta_modelo = os.path.join(pasta_saida, nome_pasta)
        os.makedirs(pasta_modelo, exist_ok=True)

        caminho_modelo = os.path.join(pasta_modelo, "modelo.joblib")
        caminho_metadata = os.path.join(pasta_modelo, "metadata.json")
        caminho_predicoes = os.path.join(pasta_modelo, "predicoes_teste.xlsx")

        joblib.dump(melhor["pipeline"], caminho_modelo)

        metadata = {
            "rodovia_identificacao": rodovia,
            "features": features,
            "target": "alvo_mais",
            "classe_1": "Mais de X",
            "classe_0": "Até X",
            "split": "temporal_80_20",
            "melhor_modelo": melhor["nome_modelo"],
            "metricas_teste": melhor["metricas"],
            "quantidades": {
                "total_bruto": int(n_total_bruto),
                "total_modelagem": int(n),
                "treino": int(len(treino)),
                "teste": int(len(teste))
            }
        }

        with open(caminho_metadata, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=4)

        predicoes = teste.copy()
        predicoes["prob_mais_modelo"] = melhor["prob"]
        predicoes["pred_mais"] = (predicoes["prob_mais_modelo"] >= 0.5).astype(int)
        predicoes["classe_real"] = np.where(predicoes["alvo_mais"] == 1, "Mais de X", "Até X")
        predicoes["classe_prevista"] = np.where(predicoes["pred_mais"] == 1, "Mais de X", "Até X")
        predicoes["acertou"] = (predicoes["alvo_mais"] == predicoes["pred_mais"]).astype(int)
        predicoes["modelo_escolhido"] = melhor["nome_modelo"]
        predicoes.to_excel(caminho_predicoes, index=False)

        resumo_final.append({
            "rodovia_identificacao": rodovia,
            "pasta_modelo": pasta_modelo,
            "modelo_escolhido": melhor["nome_modelo"],
            "qtd_total_bruto": n_total_bruto,
            "qtd_total_modelagem": n,
            "qtd_treino": len(treino),
            "qtd_teste": len(teste),
            **melhor["metricas"]
        })

        print(f"[OK] Melhor modelo salvo para: {rodovia} | Modelo: {melhor['nome_modelo']}")

    df_resumo = pd.DataFrame(resumo_final).sort_values(
        ["macro_f1", "accuracy", "ganho_vs_baseline", "auc"],
        ascending=False
    )

    df_comparativo = pd.DataFrame(comparativo_modelos).sort_values(
        ["rodovia_identificacao", "macro_f1", "accuracy", "ganho_vs_baseline", "auc"],
        ascending=[True, False, False, False, False]
    )

    caminho_resumo = os.path.join(pasta_saida, "resumo_modelos.xlsx")
    caminho_comparativo = os.path.join(pasta_saida, "comparativo_modelos.xlsx")

    df_resumo.to_excel(caminho_resumo, index=False)
    df_comparativo.to_excel(caminho_comparativo, index=False)

    print(f"\nResumo salvo em: {caminho_resumo}")
    print(f"Comparativo salvo em: {caminho_comparativo}")
    print("\nResumo final:")
    print(df_resumo)

    return df_resumo, df_comparativo


if __name__ == "__main__":
    caminho_excel = os.path.join("DadosRodovias", "dados_todas_rodovias.xlsx")

    treinar_modelos_por_rodovia_escolhendo_melhor(
        caminho_excel=caminho_excel,
        pasta_saida="modelos_rodovias_best",
        min_registros_por_rodovia=60
    )