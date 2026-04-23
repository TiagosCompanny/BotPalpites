import argparse
from datetime import datetime
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


def normalizar_colunas(df):
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    return df


def encontrar_coluna(df, candidatos):
    cols = {c.lower(): c for c in df.columns}
    for cand in candidatos:
        if cand.lower() in cols:
            return cols[cand.lower()]
    for c in df.columns:
        cl = c.lower()
        for cand in candidatos:
            if cand.lower() in cl:
                return c
    return None


def parse_resultado(valor):
    if pd.isna(valor):
        return None
    s = str(valor).strip().lower()
    if "mais" in s:
        return "mais"
    if "até" in s or "ate" in s:
        return "ate"
    return None


def carregar_e_preparar(excel_path):
    print("Lendo planilha...")
    df = pd.read_excel(excel_path)
    df = normalizar_colunas(df)

    col_abertura = encontrar_coluna(df, ["abertura", "data_abertura", "datetime_abertura"])
    col_meta = encontrar_coluna(df, ["meta_referencia", "meta", "linha", "valor"])
    col_resultado = encontrar_coluna(df, ["resultado_vencedor", "resultado", "vencedor"])

    if not col_abertura:
        raise ValueError("Não encontrei a coluna de abertura.")
    if not col_meta:
        raise ValueError("Não encontrei a coluna meta_referencia.")
    if not col_resultado:
        raise ValueError("Não encontrei a coluna resultado_vencedor.")

    df[col_abertura] = pd.to_datetime(df[col_abertura], errors="coerce")
    df[col_meta] = pd.to_numeric(df[col_meta], errors="coerce")
    df["target"] = df[col_resultado].apply(parse_resultado)

    df = df.dropna(subset=[col_abertura, col_meta, "target"]).copy()

    df["hora"] = df[col_abertura].dt.hour
    df["minuto"] = df[col_abertura].dt.minute
    df["dia_semana"] = df[col_abertura].dt.weekday
    df["dia_mes"] = df[col_abertura].dt.day
    df["mes"] = df[col_abertura].dt.month

    X = df[[col_meta, "hora", "minuto", "dia_semana", "dia_mes", "mes"]].copy()
    X = X.rename(columns={col_meta: "meta_referencia"})
    y = df["target"].map({"ate": 0, "mais": 1})

    print(f"Registros usados: {len(X)}")
    return X, y


def treinar_modelo(excel_path, modelo_path):
    print("Iniciando treino...")
    X, y = carregar_e_preparar(excel_path)

    numeric_features = ["meta_referencia", "hora", "minuto", "dia_mes", "mes"]
    categorical_features = ["dia_semana"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline([
                ("imputer", SimpleImputer(strategy="median"))
            ]), numeric_features),
            ("cat", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore"))
            ]), categorical_features),
        ]
    )

    model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(
            n_estimators=300,
            max_depth=10,
            min_samples_leaf=3,
            random_state=42
        ))
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    print("Treinando modelo...")
    model.fit(X_train, y_train)

    pred = model.predict(X_test)
    prob = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, pred)
    try:
        auc = roc_auc_score(y_test, prob)
    except Exception:
        auc = None

    print("=== METRICAS ===")
    print(f"Acuracia: {acc:.4f}")
    if auc is not None:
        print(f"ROC AUC: {auc:.4f}")
    print()
    print(classification_report(y_test, pred, target_names=["ate", "mais"]))

    payload = {"model": model}
    joblib.dump(payload, modelo_path)
    print(f"Modelo salvo em: {modelo_path}")


def prever(modelo_path, valor, hora=None, minuto=None, dia_semana=None, dia_mes=None, mes=None):
    print("Carregando modelo...")
    payload = joblib.load(modelo_path)
    model = payload["model"]

    agora = datetime.now()

    if hora is None:
        hora = agora.hour
    if minuto is None:
        minuto = agora.minute
    if dia_semana is None:
        dia_semana = agora.weekday()
    if dia_mes is None:
        dia_mes = agora.day
    if mes is None:
        mes = agora.month

    X_new = pd.DataFrame([{
        "meta_referencia": valor,
        "hora": hora,
        "minuto": minuto,
        "dia_semana": dia_semana,
        "dia_mes": dia_mes,
        "mes": mes
    }])

    proba = model.predict_proba(X_new)[0]
    pred = model.predict(X_new)[0]

    prob_ate = float(proba[0])
    prob_mais = float(proba[1])
    classe = "mais" if pred == 1 else "ate"

    print("=== PREVISAO ===")
    print(f"Data/hora usada: {dia_mes:02d}/{mes:02d} {hora:02d}:{minuto:02d}")
    print(f"Valor testado: {valor}")
    print(f"Classe prevista: {classe}")
    print(f"Probabilidade de MAIS: {prob_mais:.2%}")
    print(f"Probabilidade de ATE: {prob_ate:.2%}")


def main():
    print("Script iniciado")
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="comando", required=True)

    p_treinar = sub.add_parser("treinar")
    p_treinar.add_argument("--excel", required=True)
    p_treinar.add_argument("--modelo", required=True)

    p_prever = sub.add_parser("prever")
    p_prever.add_argument("--modelo", required=True)
    p_prever.add_argument("--valor", type=float, required=True)
    p_prever.add_argument("--hora", type=int, required=False)
    p_prever.add_argument("--minuto", type=int, required=False)
    p_prever.add_argument("--dia-semana", type=int, required=False)
    p_prever.add_argument("--dia-mes", type=int, required=False)
    p_prever.add_argument("--mes", type=int, required=False)

    args = parser.parse_args()

    if args.comando == "treinar":
        treinar_modelo(args.excel, args.modelo)
    elif args.comando == "prever":
        prever(
            args.modelo,
            args.valor,
            args.hora,
            args.minuto,
            args.dia_semana,
            args.dia_mes,
            args.mes
        )


if __name__ == "__main__":
    main()