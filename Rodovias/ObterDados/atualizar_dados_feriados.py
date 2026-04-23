import os
import time
import requests
import pandas as pd
from pathlib import Path
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

BASE_DIR = Path(__file__).resolve().parent
ARQUIVO_BASE = BASE_DIR.parent / "DadosRodovias" / "dados_todas_rodovias.xlsx"

COLUNAS_FERIADO = [
    "eh_feriado",
    "nome_feriado",
    "tipo_feriado",
    "eh_vespera_feriado",
    "eh_pos_feriado",
    "dias_para_proximo_feriado",
    "dias_desde_ultimo_feriado"
]


def criar_sessao():
    session = requests.Session()

    retry = Retry(
        total=5,
        connect=5,
        read=5,
        backoff_factor=2,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"]
    )

    adapter = HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)

    return session


session = criar_sessao()


def garantir_colunas_feriado(df):
    for coluna in COLUNAS_FERIADO:
        if coluna not in df.columns:
            df[coluna] = pd.NA
    return df


def buscar_feriados_ano(ano):
    url = f"https://brasilapi.com.br/api/feriados/v1/{ano}"

    response = session.get(url, timeout=60)
    response.raise_for_status()

    dados = response.json()
    df_feriados = pd.DataFrame(dados)

    if df_feriados.empty:
        return pd.DataFrame(columns=["date", "name", "type"])

    df_feriados["date"] = pd.to_datetime(df_feriados["date"], errors="coerce").dt.normalize()
    return df_feriados


def montar_base_feriados(anos):
    lista = []

    for i, ano in enumerate(sorted(set(anos)), start=1):
        try:
            print(f"Consultando feriados do ano {ano} ({i}/{len(set(anos))})")
            df_ano = buscar_feriados_ano(ano)

            if not df_ano.empty:
                lista.append(df_ano)

            time.sleep(0.5)

        except Exception as e:
            print(f"Erro ao consultar feriados do ano {ano}: {e}")

    if not lista:
        return pd.DataFrame(columns=["date", "name", "type"])

    df_feriados = pd.concat(lista, ignore_index=True)
    df_feriados = df_feriados.drop_duplicates(subset=["date"]).sort_values("date").reset_index(drop=True)
    return df_feriados


def preencher_colunas_feriado(df, df_feriados, coluna_data="abertura"):
    df = df.copy()

    df[coluna_data] = pd.to_datetime(df[coluna_data], errors="coerce")
    df["data_dia"] = df[coluna_data].dt.normalize()

    mapa_nome = dict(zip(df_feriados["date"], df_feriados["name"]))
    mapa_tipo = dict(zip(df_feriados["date"], df_feriados["type"]))
    datas_feriado = set(df_feriados["date"].dropna())

    df["eh_feriado_novo"] = df["data_dia"].isin(datas_feriado).astype(int)
    df["nome_feriado_novo"] = df["data_dia"].map(mapa_nome)
    df["tipo_feriado_novo"] = df["data_dia"].map(mapa_tipo)

    datas_ordenadas = sorted(datas_feriado)

    if datas_ordenadas:
        serie_feriados = pd.Series(pd.to_datetime(datas_ordenadas))

        df["eh_vespera_feriado_novo"] = (
            (df["data_dia"] + pd.Timedelta(days=1)).isin(datas_feriado)
        ).astype(int)

        df["eh_pos_feriado_novo"] = (
            (df["data_dia"] - pd.Timedelta(days=1)).isin(datas_feriado)
        ).astype(int)

        def dias_para_proximo(data):
            futuros = serie_feriados[serie_feriados >= data]
            if futuros.empty:
                return pd.NA
            return int((futuros.iloc[0] - data).days)

        def dias_desde_ultimo(data):
            passados = serie_feriados[serie_feriados <= data]
            if passados.empty:
                return pd.NA
            return int((data - passados.iloc[-1]).days)

        df["dias_para_proximo_feriado_novo"] = df["data_dia"].apply(dias_para_proximo)
        df["dias_desde_ultimo_feriado_novo"] = df["data_dia"].apply(dias_desde_ultimo)
    else:
        df["eh_vespera_feriado_novo"] = pd.NA
        df["eh_pos_feriado_novo"] = pd.NA
        df["dias_para_proximo_feriado_novo"] = pd.NA
        df["dias_desde_ultimo_feriado_novo"] = pd.NA

    return df


def atualizar_dados_feriados(caminho_arquivo, coluna_data="abertura"):
    if not os.path.exists(caminho_arquivo):
        print(f"Arquivo não encontrado: {caminho_arquivo}")
        return

    df = pd.read_excel(caminho_arquivo)
    print(f"Linhas carregadas: {len(df)}")

    if df.empty:
        print("Planilha vazia.")
        return

    if coluna_data not in df.columns:
        print(f"Coluna de data não encontrada: {coluna_data}")
        return

    df = garantir_colunas_feriado(df)

    df[coluna_data] = pd.to_datetime(df[coluna_data], errors="coerce")
    df_validas = df[df[coluna_data].notna()].copy()

    if df_validas.empty:
        print("Nenhuma linha com data válida.")
        return

    anos = df_validas[coluna_data].dt.year.dropna().astype(int).unique().tolist()
    print(f"Anos encontrados na base: {sorted(anos)}")

    df_feriados = montar_base_feriados(anos)

    if df_feriados.empty:
        print("Nenhum feriado retornado pela API.")
        return

    df_preenchido = preencher_colunas_feriado(df, df_feriados, coluna_data=coluna_data)

    mapeamento = {
        "eh_feriado": "eh_feriado_novo",
        "nome_feriado": "nome_feriado_novo",
        "tipo_feriado": "tipo_feriado_novo",
        "eh_vespera_feriado": "eh_vespera_feriado_novo",
        "eh_pos_feriado": "eh_pos_feriado_novo",
        "dias_para_proximo_feriado": "dias_para_proximo_feriado_novo",
        "dias_desde_ultimo_feriado": "dias_desde_ultimo_feriado_novo"
    }

    total_atualizadas = 0

    for coluna_antiga, coluna_nova in mapeamento.items():
        if coluna_nova not in df_preenchido.columns:
            continue

        mascara_pendente = df[coluna_antiga].isna()
        qtd = int(mascara_pendente.sum())

        if qtd > 0:
            df.loc[mascara_pendente, coluna_antiga] = df_preenchido.loc[mascara_pendente, coluna_nova]
            total_atualizadas += qtd

    if "fechamento" in df.columns:
        df["fechamento"] = pd.to_datetime(df["fechamento"], errors="coerce")
        df = df.sort_values(by="fechamento", ascending=False)

    colunas_auxiliares = [
        "data_dia",
        "eh_feriado_novo",
        "nome_feriado_novo",
        "tipo_feriado_novo",
        "eh_vespera_feriado_novo",
        "eh_pos_feriado_novo",
        "dias_para_proximo_feriado_novo",
        "dias_desde_ultimo_feriado_novo"
    ]

    df = df.drop(columns=colunas_auxiliares, errors="ignore")
    df.to_excel(caminho_arquivo, index=False)

    print(f"Campos preenchidos de feriado: {total_atualizadas}")
    print(f"Planilha atualizada com sucesso: {caminho_arquivo}")


atualizar_dados_feriados(ARQUIVO_BASE, coluna_data="abertura")