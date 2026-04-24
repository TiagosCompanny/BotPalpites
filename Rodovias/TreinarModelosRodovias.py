import os
import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)
from sklearn.ensemble import RandomForestClassifier
from Utils import LogService, TreinamentoLog


BASE_DIR = Path(__file__).resolve().parent
ARQUIVO_BASE = BASE_DIR / "DadosRodovias" / "dados_todas_rodovias.xlsx"
MODELOS_DIR = BASE_DIR / "modelos_rodovias"

CATALOGO_RODOVIAS = {
    "braganca_paulista": {
        "rodovia_identificacao": "Rodovia Arão Sahm, KM 95 — Bragança Paulista (SP).",
        "pasta_saida": str(
            MODELOS_DIR / "rodovia_arao_sahm_km_95_braganca_paulista_sp_rf_regimes"
        ),
        "filtro_rodovia_legado": "Rodovia Arão Sahm",
        "nome_exibicao": "BRAGANÇA PAULISTA",
    },
    "caraguatatuba": {
        "rodovia_identificacao": "Doutor Manoel Hyppolito Rego, KM 83 — Caraguatatuba (SP).",
        "pasta_saida": str(
            MODELOS_DIR / "doutor_manoel_hyppolito_rego_km_83_caraguatatuba_sp_rf_regimes"
        ),
        "filtro_rodovia_legado": "Doutor Manoel Hyppolito Rego",
        "nome_exibicao": "CARAGUATATUBA",
    },
    "pindamonhangaba": {
        "rodovia_identificacao": "Floriano Rodrigues Pinheiro, KM 26 — Pindamonhangaba (SP).",
        "pasta_saida": str(
            MODELOS_DIR / "floriano_rodrigues_pinheiro_km_26_pindamonhangaba_sp_rf_regimes"
        ),
        "filtro_rodovia_legado": "Floriano Rodrigues Pinheiro",
        "nome_exibicao": "PINDAMONHANGABA",
    },
}

FEATURES_PRODUCAO = [
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
    "bloco_madrugada",
    "bloco_pico_manha",
    "bloco_manha",
    "bloco_almoco",
    "bloco_tarde_pico",
    "bloco_noite",
    "bloco_late_night",
    "bloco_outro",
    "interacao_meta_hora_sin",
    "interacao_meta_hora_cos",
    "interacao_meta_temperatura",
    "interacao_meta_umidade",
]

log_service = LogService()


def parse_float(x):
    try:
        return float(str(x).replace(",", "."))
    except Exception:
        return np.nan


def obter_config_rodovia(rodovia_chave):
    if rodovia_chave not in CATALOGO_RODOVIAS:
        chaves_disponiveis = ", ".join(CATALOGO_RODOVIAS.keys())
        raise ValueError(
            f"Rodovia '{rodovia_chave}' não encontrada no catálogo. Disponíveis: {chaves_disponiveis}"
        )
    return CATALOGO_RODOVIAS[rodovia_chave]


def preparar_base(df, config):
    df = df.copy()

    for col in ["abertura", "fechamento", "resolvido_em"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    df = df.dropna(
        subset=["rodovia_identificacao", "abertura", "resultado_vencedor"]
    ).copy()

    rodovia_identificacao = config["rodovia_identificacao"]
    filtro_rodovia_legado = config.get("filtro_rodovia_legado")

    if "rodovia_identificacao" in df.columns:
        df = df[df["rodovia_identificacao"] == rodovia_identificacao].copy()
    elif "rodovia" in df.columns and filtro_rodovia_legado:
        df = df[
            df["rodovia"].astype(str).str.contains(
                filtro_rodovia_legado, case=False, na=False
            )
        ].copy()

    if df.empty and "rodovia" in df.columns and filtro_rodovia_legado:
        df = df[
            df["rodovia"].astype(str).str.contains(
                filtro_rodovia_legado, case=False, na=False
            )
        ].copy()

    if df.empty:
        raise ValueError(
            f"Nenhum registro encontrado para a rodovia: {rodovia_identificacao}"
        )

    df = df.sort_values("abertura").reset_index(drop=True)

    df["alvo_mais"] = (
        df["resultado_vencedor"].astype(str).str.contains("Mais", case=False, na=False)
    ).astype(int)

    for col in [
        "meta_referencia",
        "temperatura_2m",
        "umidade_relativa",
        "chuva_mm",
        "cobertura_nuvens",
        "estava_chovendo",
    ]:
        if col in df.columns:
            df[col] = df[col].apply(parse_float)

    if "estava_chovendo" not in df.columns:
        df["estava_chovendo"] = 0.0
    else:
        df["estava_chovendo"] = df["estava_chovendo"].fillna(0.0)

    df["meta_num"] = df["meta_referencia"].apply(parse_float)

    df["hora"] = df["abertura"].dt.hour
    df["minuto"] = df["abertura"].dt.minute
    df["dia_semana"] = df["abertura"].dt.dayofweek
    df["mes"] = df["abertura"].dt.month
    df["hora_decimal"] = df["hora"] + (df["minuto"] / 60.0)

    df["fim_semana"] = df["dia_semana"].isin([5, 6]).astype(int)

    df["hora_sin"] = np.sin(2 * np.pi * df["hora_decimal"] / 24)
    df["hora_cos"] = np.cos(2 * np.pi * df["hora_decimal"] / 24)
    df["dia_semana_sin"] = np.sin(2 * np.pi * df["dia_semana"] / 7)
    df["dia_semana_cos"] = np.cos(2 * np.pi * df["dia_semana"] / 7)
    df["mes_sin"] = np.sin(2 * np.pi * (df["mes"] - 1) / 12)
    df["mes_cos"] = np.cos(2 * np.pi * (df["mes"] - 1) / 12)

    df["eh_pico_manha"] = df["hora"].between(5, 8).astype(int)
    df["eh_pico_tarde"] = df["hora"].between(16, 18).astype(int)
    df["eh_sexta"] = (df["dia_semana"] == 4).astype(int)
    df["eh_domingo"] = (df["dia_semana"] == 6).astype(int)
    df["eh_sabado"] = (df["dia_semana"] == 5).astype(int)
    df["eh_dia_util"] = (df["dia_semana"] <= 4).astype(int)

    condicoes = [
        (df["hora_decimal"] >= 0) & (df["hora_decimal"] < 5),
        (df["hora_decimal"] >= 5) & (df["hora_decimal"] < 8),
        (df["hora_decimal"] >= 8) & (df["hora_decimal"] < 11),
        (df["hora_decimal"] >= 11) & (df["hora_decimal"] < 14),
        (df["hora_decimal"] >= 14) & (df["hora_decimal"] < 18),
        (df["hora_decimal"] >= 18) & (df["hora_decimal"] < 22),
        (df["hora_decimal"] >= 22) & (df["hora_decimal"] <= 23.999999),
    ]
    valores = [
        "madrugada",
        "pico_manha",
        "manha",
        "almoco",
        "tarde_pico",
        "noite",
        "late_night",
    ]

    df["bloco_dia"] = np.select(condicoes, valores, default="outro")
    df["bloco_dia_cod"] = pd.Categorical(
        df["bloco_dia"],
        categories=[
            "madrugada",
            "pico_manha",
            "manha",
            "almoco",
            "tarde_pico",
            "noite",
            "late_night",
            "outro",
        ],
    ).codes

    dummies = pd.get_dummies(df["bloco_dia"], prefix="bloco")
    df = pd.concat([df, dummies], axis=1)

    for col in [
        "bloco_madrugada",
        "bloco_pico_manha",
        "bloco_manha",
        "bloco_almoco",
        "bloco_tarde_pico",
        "bloco_noite",
        "bloco_late_night",
        "bloco_outro",
    ]:
        if col not in df.columns:
            df[col] = 0

    df["interacao_meta_hora_sin"] = df["meta_num"] * df["hora_sin"]
    df["interacao_meta_hora_cos"] = df["meta_num"] * df["hora_cos"]
    df["interacao_meta_temperatura"] = df["meta_num"] * df["temperatura_2m"]
    df["interacao_meta_umidade"] = df["meta_num"] * df["umidade_relativa"]

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
        "precision_mais": round(
            float(precision_score(y_true, pred, zero_division=0)), 6
        ),
        "recall_mais": round(float(recall_score(y_true, pred, zero_division=0)), 6),
        "f1_mais": round(float(f1_score(y_true, pred, zero_division=0)), 6),
        "f1_ate": round(
            float(f1_score(y_true, pred, pos_label=0, zero_division=0)), 6
        ),
        "macro_f1": round(
            float(f1_score(y_true, pred, average="macro", zero_division=0)), 6
        ),
        "baseline_majoritaria": round(float(baseline), 6),
        "ganho_vs_baseline": round(float(acc - baseline), 6),
        "taxa_mais_teste": round(float(y_true.mean()), 6),
    }


def treinar_modelo_rodovia(rodovia_chave, caminho_excel=ARQUIVO_BASE):
    config = obter_config_rodovia(rodovia_chave)

    rodovia_identificacao = config["rodovia_identificacao"]
    pasta_saida = config["pasta_saida"]
    nome_exibicao = config.get("nome_exibicao", rodovia_identificacao)

    os.makedirs(pasta_saida, exist_ok=True)

    df = pd.read_excel(caminho_excel)
    df = preparar_base(df, config)

    features = [c for c in FEATURES_PRODUCAO if c in df.columns]

    df_model = df.dropna(subset=["alvo_mais", "meta_num"]).copy()

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

    X_treino = treino[features]
    y_treino = treino["alvo_mais"]

    X_teste = teste[features]
    y_teste = teste["alvo_mais"]

    modelo = Pipeline(
        [
            ("imp", SimpleImputer(strategy="median")),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=900,
                    max_depth=16,
                    min_samples_split=4,
                    min_samples_leaf=2,
                    max_features="sqrt",
                    class_weight="balanced_subsample",
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )

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
    print(
        classification_report(
            y_teste, pred, target_names=["Até a Meta (0)", "Acima da Meta (1)"]
        )
    )
    print("Matriz de Confusão:")
    print(confusion_matrix(y_teste, pred))

    modelo_final = modelo.named_steps["model"]
    importancias = pd.DataFrame(
        {"variavel": features, "importancia": modelo_final.feature_importances_}
    ).sort_values(by="importancia", ascending=False)

    print("\n=== IMPORTÂNCIA DAS VARIÁVEIS ===")
    print(importancias.head(30))

    caminho_modelo = os.path.join(pasta_saida, "modelo.joblib")
    caminho_metadata = os.path.join(pasta_saida, "metadata.json")
    caminho_predicoes = os.path.join(pasta_saida, "predicoes_teste.xlsx")
    caminho_importancias = os.path.join(pasta_saida, "importancias.xlsx")

    bundle = {
        "modelo": modelo,
        "features": features,
        "rodovia_identificacao": rodovia_identificacao,
        "rodovia_chave": rodovia_chave,
    }
    joblib.dump(bundle, caminho_modelo)

    metadata = {
        "rodovia_chave": rodovia_chave,
        "rodovia_identificacao": rodovia_identificacao,
        "features": features,
        "target": "alvo_mais",
        "classe_1": "Mais de X",
        "classe_0": "Até X",
        "modelo_escolhido": "rf_producao_friendly_sem_lags",
        "split": "temporal_80_20",
        "metricas_teste": metricas,
        "quantidades": {
            "total_modelagem": int(n),
            "treino": int(len(treino)),
            "teste": int(len(teste)),
        },
    }

    with open(caminho_metadata, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=4)

    predicoes = teste.copy()
    predicoes["prob_mais_modelo"] = prob
    predicoes["pred_mais"] = pred
    predicoes["classe_real"] = np.where(predicoes["alvo_mais"] == 1, "Mais de X", "Até X")
    predicoes["classe_prevista"] = np.where(
        predicoes["pred_mais"] == 1, "Mais de X", "Até X"
    )
    predicoes["acertou"] = (predicoes["alvo_mais"] == predicoes["pred_mais"]).astype(int)

    predicoes.to_excel(caminho_predicoes, index=False)
    importancias.to_excel(caminho_importancias, index=False)

    print(f"\nModelo salvo em: {caminho_modelo}")
    print(f"Metadata salva em: {caminho_metadata}")
    print(f"Predições salvas em: {caminho_predicoes}")
    print(f"Importâncias salvas em: {caminho_importancias}")

    log_service.registrar_treinamento(
        TreinamentoLog(
            data_hora=LogService.agora_str(),
            rodovia=rodovia_identificacao,
            nome_modelo="rf_producao_friendly_sem_lags",
            versao_modelo="v1",
            algoritmo="RandomForestClassifier",
            arquivo_base=str(caminho_excel),
            quantidade_registros=int(n),
            quantidade_treino=int(len(treino)),
            quantidade_teste=int(len(teste)),
            campos_utilizados=",".join(features),
            campo_alvo="alvo_mais",
            balanceamento_aplicado="class_weight=balanced_subsample",
            acuracia=float(metricas["accuracy"]),
            macro_f1=float(metricas["macro_f1"]),
            f1_mais=float(metricas["f1_mais"]),
            f1_ate=float(metricas["f1_ate"]),
            baseline=float(metricas["baseline_majoritaria"]),
        )
    )


if __name__ == "__main__":
    for rodovia_chave in CATALOGO_RODOVIAS.keys():
        print(f"\n\n==============================")
        print(f"INICIANDO MODELAGEM PARA: {rodovia_chave}")
        print(f"==============================\n")

        try:
            treinar_modelo_rodovia(
                rodovia_chave=rodovia_chave,
                caminho_excel=ARQUIVO_BASE,
            )
        except Exception as e:
            print(f"Erro ao modelar rodovia '{rodovia_chave}': {e}")
            config_erro = CATALOGO_RODOVIAS.get(rodovia_chave, {})
            log_service.registrar_treinamento(
                TreinamentoLog(
                    data_hora=LogService.agora_str(),
                    rodovia=config_erro.get("rodovia_identificacao", rodovia_chave),
                    nome_modelo="rf_producao_friendly_sem_lags",
                    versao_modelo="v1",
                    algoritmo="RandomForestClassifier",
                    arquivo_base=str(ARQUIVO_BASE),
                    quantidade_registros=0,
                    quantidade_treino=0,
                    quantidade_teste=0,
                    campos_utilizados="",
                    campo_alvo="alvo_mais",
                    balanceamento_aplicado=f"erro: {e}",
                    acuracia=0.0,
                    macro_f1=0.0,
                    f1_mais=0.0,
                    f1_ate=0.0,
                    baseline=0.0,
                )
            )