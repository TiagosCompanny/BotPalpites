import os
import re
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder


def normalizar_nome_arquivo(texto):
    texto = str(texto).strip().lower()
    texto = re.sub(r'[^\w\s-]', '', texto, flags=re.UNICODE)
    texto = re.sub(r'[\s]+', '_', texto)
    return texto


def parse_meta(x):
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
    df["hora"] = df["abertura"].dt.hour
    df["minuto"] = df["abertura"].dt.minute
    df["dow"] = df["abertura"].dt.dayofweek
    df["mes"] = df["abertura"].dt.month
    df["meta_num"] = df["meta_referencia"].apply(parse_meta)

    return df


def criar_pipeline_exato():
    pre = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline([
                    ("imp", SimpleImputer(strategy="median"))
                ]),
                ["minuto", "meta_num"]
            ),
            (
                "cat",
                Pipeline([
                    ("imp", SimpleImputer(strategy="most_frequent")),
                    ("ohe", OneHotEncoder(handle_unknown="ignore"))
                ]),
                ["hora", "dow", "mes"]
            )
        ]
    )

    modelo = Pipeline([
        ("pre", pre),
        ("model", RandomForestClassifier(
            n_estimators=400,
            random_state=42,
            class_weight="balanced_subsample",
            min_samples_leaf=4,
            max_depth=10
        ))
    ])

    return modelo


def treinar_modelos_exatamente_como_estudo(
    caminho_excel,
    pasta_saida="modelos_rodovias_exato",
    min_registros_por_rodovia=60
):
    os.makedirs(pasta_saida, exist_ok=True)

    df = pd.read_excel(caminho_excel)
    df = preparar_base(df)

    resumo = []

    for rodovia, g in df.sort_values("abertura").groupby("rodovia_identificacao"):
        g = g.sort_values("abertura").reset_index(drop=True).copy()
        n = len(g)

        if n < min_registros_por_rodovia:
            print(f"[IGNORADO] {rodovia} | poucos registros: {n}")
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

        features = ["hora", "minuto", "dow", "mes", "meta_num"]

        X_treino = treino[features]
        y_treino = treino["alvo_mais"]

        X_teste = teste[features]
        y_teste = teste["alvo_mais"]

        modelo = criar_pipeline_exato()
        modelo.fit(X_treino, y_treino)

        prob = modelo.predict_proba(X_teste)[:, 1]
        pred = (prob >= 0.5).astype(int)

        acc = accuracy_score(y_teste, pred)
        prec = precision_score(y_teste, pred, zero_division=0)
        rec = recall_score(y_teste, pred, zero_division=0)
        f1 = f1_score(y_teste, pred, zero_division=0)
        auc = roc_auc_score(y_teste, prob)
        baseline = max(y_teste.mean(), 1 - y_teste.mean())

        nome_pasta = normalizar_nome_arquivo(rodovia)
        pasta_modelo = os.path.join(pasta_saida, nome_pasta)
        os.makedirs(pasta_modelo, exist_ok=True)

        caminho_modelo = os.path.join(pasta_modelo, "modelo.joblib")
        caminho_metadata = os.path.join(pasta_modelo, "metadata.json")
        caminho_predicoes = os.path.join(pasta_modelo, "predicoes_teste.xlsx")

        joblib.dump(modelo, caminho_modelo)

        metadata = {
            "rodovia_identificacao": rodovia,
            "features": features,
            "target": "alvo_mais",
            "classe_1": "Mais de X",
            "classe_0": "Até X",
            "split": "temporal_80_20",
            "modelo": {
                "tipo": "RandomForestClassifier",
                "n_estimators": 400,
                "random_state": 42,
                "class_weight": "balanced_subsample",
                "min_samples_leaf": 4,
                "max_depth": 10
            },
            "metricas_teste": {
                "accuracy": round(float(acc), 6),
                "auc": round(float(auc), 6),
                "precision_mais": round(float(prec), 6),
                "recall_mais": round(float(rec), 6),
                "f1_mais": round(float(f1), 6),
                "baseline_majoritaria": round(float(baseline), 6),
                "ganho_vs_baseline": round(float(acc - baseline), 6),
                "taxa_mais_teste": round(float(y_teste.mean()), 6)
            },
            "quantidades": {
                "total": int(n),
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

        resumo.append({
            "rodovia_identificacao": rodovia,
            "pasta_modelo": pasta_modelo,
            "qtd_total": n,
            "qtd_treino": len(treino),
            "qtd_teste": len(teste),
            "accuracy": round(float(acc), 6),
            "auc": round(float(auc), 6),
            "precision_mais": round(float(prec), 6),
            "recall_mais": round(float(rec), 6),
            "f1_mais": round(float(f1), 6),
            "baseline_majoritaria": round(float(baseline), 6),
            "ganho_vs_baseline": round(float(acc - baseline), 6),
            "taxa_mais_teste": round(float(y_teste.mean()), 6)
        })

        print(f"[OK] Modelo salvo para: {rodovia}")

    df_resumo = pd.DataFrame(resumo).sort_values("accuracy", ascending=False)
    caminho_resumo = os.path.join(pasta_saida, "resumo_modelos.xlsx")
    df_resumo.to_excel(caminho_resumo, index=False)

    print(f"\nResumo salvo em: {caminho_resumo}")
    print("\nResumo final:")
    print(df_resumo)

    return df_resumo


if __name__ == "__main__":
    caminho_excel = os.path.join("DadosRodovias", "dados_todas_rodovias.xlsx")

    treinar_modelos_exatamente_como_estudo(
        caminho_excel=caminho_excel,
        pasta_saida="modelos_rodovias_exato",
        min_registros_por_rodovia=60
    )