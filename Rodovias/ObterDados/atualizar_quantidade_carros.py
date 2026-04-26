import os
import json
import time
import requests
import pandas as pd
from pathlib import Path
from datetime import datetime
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


BASE_DIR = Path(__file__).resolve().parent
DADOS_DIR = BASE_DIR.parent / "DadosRodovias"

ARQUIVO_BASE = DADOS_DIR / "dados_todas_rodovias.xlsx"
ARQUIVO_PASSAGENS = DADOS_DIR / "passagens_carros_por_mercado.xlsx"
ARQUIVO_ERROS = DADOS_DIR / "erros_atualizar_passagens_carros_por_mercado.xlsx"

COLUNA_ID_MERCADO_BASE = "id"
COLUNA_ID_MERCADO_PASSAGENS = "MercadoId"

SALVAR_A_CADA = 200
PAUSA_ENTRE_MERCADOS = 0.3


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


def normalizar_id(valor):
    if pd.isna(valor):
        return None

    texto = str(valor).strip()

    if texto == "":
        return None

    try:
        numero = float(texto)
        if numero.is_integer():
            return str(int(numero))
    except Exception:
        pass

    return texto


def montar_url_mercado(market_id):
    market_id = int(float(str(market_id).strip()))
    return f"https://app.palpita.io/live/{market_id}-market/rodovia-5-minutos-qu-{market_id}"


def obter_html_mercado(market_id):
    url = montar_url_mercado(market_id)

    headers = {
        "User-Agent": "Mozilla/5.0"
    }

    response = session.get(url, headers=headers, timeout=60)
    response.raise_for_status()

    return response.text


def normalizar_html(html):
    return (
        html
        .replace('\\"', '"')
        .replace("\\/", "/")
        .replace("&quot;", '"')
        .replace("\\u003c", "<")
        .replace("\\u003e", ">")
        .replace("\\u0026", "&")
    )


def extrair_array_json_balanceado(texto, nome_campo):
    texto = normalizar_html(texto)

    pos = texto.find(f'"{nome_campo}"')

    if pos == -1:
        return None

    inicio_array = texto.find("[", pos)

    if inicio_array == -1:
        return None

    nivel = 0
    dentro_string = False
    escape = False

    for i in range(inicio_array, len(texto)):
        char = texto[i]

        if escape:
            escape = False
            continue

        if char == "\\":
            escape = True
            continue

        if char == '"':
            dentro_string = not dentro_string
            continue

        if dentro_string:
            continue

        if char == "[":
            nivel += 1

        elif char == "]":
            nivel -= 1

            if nivel == 0:
                return texto[inicio_array:i + 1]

    return None


def obter_graph_data_mercado(market_id):
    html = obter_html_mercado(market_id)

    texto_array = extrair_array_json_balanceado(
        html,
        "graphData"
    )

    if not texto_array:
        return []

    try:
        return json.loads(texto_array)
    except Exception as e:
        print(f"Erro ao converter graphData do mercado {market_id}: {e}")
        return []


def converter_timestamp_para_data_hora(timestamp):
    if timestamp is None:
        return None

    try:
        return datetime.fromtimestamp(int(timestamp)).strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return None


def formatar_segundos(segundos):
    segundos = int(segundos)
    horas = segundos // 3600
    minutos = (segundos % 3600) // 60
    segs = segundos % 60
    return f"{horas:02d}:{minutos:02d}:{segs:02d}"


def carregar_ids_mercados_base():
    if not os.path.exists(ARQUIVO_BASE):
        raise FileNotFoundError(f"Arquivo base não encontrado: {ARQUIVO_BASE}")

    df_base = pd.read_excel(ARQUIVO_BASE)

    if COLUNA_ID_MERCADO_BASE not in df_base.columns:
        raise ValueError(f"Coluna '{COLUNA_ID_MERCADO_BASE}' não encontrada em {ARQUIVO_BASE}")

    ids = df_base[COLUNA_ID_MERCADO_BASE].apply(normalizar_id)
    ids = ids.dropna()
    ids = ids[ids != ""]

    return set(ids.tolist())


def carregar_passagens_existentes():
    print(f"Lendo planilha de passagens em: {ARQUIVO_PASSAGENS}")

    if not os.path.exists(ARQUIVO_PASSAGENS):
        print("Planilha de passagens não existe. Será criada.")
        return pd.DataFrame(columns=["MercadoId", "Value", "DataHora"])

    df_passagens = pd.read_excel(ARQUIVO_PASSAGENS)

    print(f"Linhas carregadas da planilha de passagens: {len(df_passagens)}")
    print(f"Colunas encontradas: {list(df_passagens.columns)}")

    if df_passagens.empty:
        return pd.DataFrame(columns=["MercadoId", "Value", "DataHora"])

    if COLUNA_ID_MERCADO_PASSAGENS not in df_passagens.columns:
        raise ValueError(f"Coluna '{COLUNA_ID_MERCADO_PASSAGENS}' não encontrada em {ARQUIVO_PASSAGENS}")

    df_passagens[COLUNA_ID_MERCADO_PASSAGENS] = df_passagens[COLUNA_ID_MERCADO_PASSAGENS].apply(normalizar_id)

    if "DataHora" in df_passagens.columns:
        df_passagens["DataHora"] = pd.to_datetime(df_passagens["DataHora"], errors="coerce")

    return df_passagens


def obter_ids_passagens_existentes(df_passagens):
    if df_passagens.empty:
        return set()

    if COLUNA_ID_MERCADO_PASSAGENS not in df_passagens.columns:
        return set()

    ids = df_passagens[COLUNA_ID_MERCADO_PASSAGENS].apply(normalizar_id)
    ids = ids.dropna()
    ids = ids[ids != ""]

    return set(ids.tolist())


def montar_registros_passagens(mercado_id, graph_data):
    registros = []

    for item in graph_data:
        registros.append({
            "MercadoId": normalizar_id(mercado_id),
            "Value": item.get("value"),
            "DataHora": converter_timestamp_para_data_hora(item.get("timestamp"))
        })

    return registros


def salvar_passagens(df_passagens):
    DADOS_DIR.mkdir(parents=True, exist_ok=True)

    if df_passagens.empty:
        df_passagens.to_excel(ARQUIVO_PASSAGENS, index=False)
        return

    df_passagens = df_passagens.copy()

    df_passagens["MercadoId"] = df_passagens["MercadoId"].apply(normalizar_id)

    if "DataHora" in df_passagens.columns:
        df_passagens["DataHora"] = pd.to_datetime(df_passagens["DataHora"], errors="coerce")

    df_passagens = df_passagens.drop_duplicates(
        subset=["MercadoId", "Value", "DataHora"],
        keep="last"
    )

    df_passagens = df_passagens.sort_values(
        by=["MercadoId", "DataHora"],
        ascending=[True, True]
    ).reset_index(drop=True)

    df_passagens.to_excel(ARQUIVO_PASSAGENS, index=False)


def salvar_erros(erros):
    if not erros:
        return

    df_erros = pd.DataFrame(erros)
    df_erros.to_excel(ARQUIVO_ERROS, index=False)


def atualizar_passagens_carros_por_mercado():
    inicio = time.time()

    print("Iniciando atualização incremental de passagens...")
    print(f"Arquivo base: {ARQUIVO_BASE}")
    print(f"Arquivo passagens: {ARQUIVO_PASSAGENS}")

    ids_mercados_base = carregar_ids_mercados_base()
    df_passagens = carregar_passagens_existentes()
    ids_passagens_existentes = obter_ids_passagens_existentes(df_passagens)

    ids_pendentes = sorted(
        ids_mercados_base - ids_passagens_existentes,
        key=lambda x: int(x) if str(x).isdigit() else str(x)
    )

    print("")
    print(f"Mercados em dados_todas_rodovias.xlsx: {len(ids_mercados_base)}")
    print(f"Mercados já existentes em passagens_carros_por_mercado.xlsx: {len(ids_passagens_existentes)}")
    print(f"Mercados pendentes para buscar passagens: {len(ids_pendentes)}")
    print("")

    if not ids_pendentes:
        print("Nenhum mercado pendente encontrado. Nada para atualizar.")
        return

    novos_registros = []
    erros = []
    total = len(ids_pendentes)

    for posicao, mercado_id in enumerate(ids_pendentes, start=1):
        try:
            graph_data = obter_graph_data_mercado(mercado_id)
            registros = montar_registros_passagens(mercado_id, graph_data)

            if registros:
                novos_registros.extend(registros)

            percentual = (posicao / total) * 100
            decorrido = time.time() - inicio
            media_por_mercado = decorrido / posicao
            restante = media_por_mercado * (total - posicao)

            print(
                f"[{posicao}/{total}] "
                f"{percentual:.2f}% | "
                f"Mercado {mercado_id} -> {len(registros)} passagens | "
                f"Novas acumuladas: {len(novos_registros)} | "
                f"Decorrido: {formatar_segundos(decorrido)} | "
                f"Restante estimado: {formatar_segundos(restante)}"
            )

        except Exception as e:
            percentual = (posicao / total) * 100
            decorrido = time.time() - inicio
            media_por_mercado = decorrido / posicao
            restante = media_por_mercado * (total - posicao)

            erros.append({
                "MercadoId": mercado_id,
                "Erro": str(e),
                "DataErro": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })

            print(
                f"[{posicao}/{total}] "
                f"{percentual:.2f}% | "
                f"Erro no mercado {mercado_id}: {e} | "
                f"Decorrido: {formatar_segundos(decorrido)} | "
                f"Restante estimado: {formatar_segundos(restante)}"
            )

        if posicao % SALVAR_A_CADA == 0:
            if novos_registros:
                df_novos = pd.DataFrame(novos_registros)
                df_passagens = pd.concat([df_passagens, df_novos], ignore_index=True)
                salvar_passagens(df_passagens)
                df_passagens = carregar_passagens_existentes()
                novos_registros = []
                print("Salvamento parcial realizado.")

            salvar_erros(erros)

        time.sleep(0.5)

    if novos_registros:
        df_novos = pd.DataFrame(novos_registros)
        df_passagens = pd.concat([df_passagens, df_novos], ignore_index=True)

    salvar_passagens(df_passagens)
    salvar_erros(erros)

    df_final = carregar_passagens_existentes()
    ids_final = obter_ids_passagens_existentes(df_final)

    tempo_total = time.time() - inicio

    print("")
    print("Atualização concluída.")
    print(f"Arquivo atualizado: {ARQUIVO_PASSAGENS}")
    print(f"Mercados na base principal: {len(ids_mercados_base)}")
    print(f"Mercados com passagens agora: {len(ids_final)}")
    print(f"Mercados ainda pendentes: {len(ids_mercados_base - ids_final)}")
    print(f"Total de linhas de passagens: {len(df_final)}")
    print(f"Erros registrados: {len(erros)}")
    print(f"Tempo total: {formatar_segundos(tempo_total)}")

    if erros:
        print(f"Arquivo de erros: {ARQUIVO_ERROS}")


if __name__ == "__main__":
    atualizar_passagens_carros_por_mercado()