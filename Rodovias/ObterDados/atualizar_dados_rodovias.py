import os
import requests
import pandas as pd
import time
import re
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
ARQUIVO_BASE = BASE_DIR.parent / "DadosRodovias" / "dados_todas_rodovias.xlsx"

def extrair_dados_rodovia(descricao):
    if not descricao or not descricao.strip():
        return {
            'rodovia_identificacao': None,
            'rodovia': None,
            'km': None,
            'cidade': None,
            'uf': None
        }

    primeira_linha = descricao.splitlines()[0].strip()
    primeira_linha = primeira_linha.lstrip('•').strip()

    padrao = r'^(.*?),\s*KM\s*(\d+(?:[.,]\d+)?)\s*[—-]\s*(.*?)\s*\(([A-Z]{2})\)\.?$'
    match = re.match(padrao, primeira_linha)

    if match:
        km = match.group(2).replace(",", ".")
        return {
            'rodovia_identificacao': primeira_linha,
            'rodovia': match.group(1).strip(),
            'km': km,
            'cidade': match.group(3).strip(),
            'uf': match.group(4).strip()
        }

    return {
        'rodovia_identificacao': primeira_linha,
        'rodovia': None,
        'km': None,
        'cidade': None,
        'uf': None
    }

def tratar_dataframe(df):
    if df.empty:
        return df

    df['abertura'] = pd.to_datetime(df['abertura'], errors='coerce').dt.tz_localize(None)
    df['fechamento'] = pd.to_datetime(df['fechamento'], errors='coerce').dt.tz_localize(None)
    df['resolvido_em'] = pd.to_datetime(df['resolvido_em'], errors='coerce').dt.tz_localize(None)

    df['km'] = pd.to_numeric(df['km'], errors='coerce')
    df['meta_referencia'] = pd.to_numeric(df['meta_referencia'], errors='coerce')
    df['prob_mais'] = pd.to_numeric(df['prob_mais'], errors='coerce')
    df['prob_ate'] = pd.to_numeric(df['prob_ate'], errors='coerce')

    df['dia_semana_num'] = df['abertura'].dt.dayofweek
    df['dia_semana'] = df['abertura'].dt.day_name()
    df['hora_abertura'] = df['abertura'].dt.hour
    df['mes'] = df['abertura'].dt.month
    df['fim_semana'] = df['dia_semana_num'].isin([5, 6]).astype(int)

    df['hora_sin'] = np.sin(2 * np.pi * df['hora_abertura'] / 24)
    df['hora_cos'] = np.cos(2 * np.pi * df['hora_abertura'] / 24)

    df['dia_semana_sin'] = np.sin(2 * np.pi * df['dia_semana_num'] / 7)
    df['dia_semana_cos'] = np.cos(2 * np.pi * df['dia_semana_num'] / 7)

    df['mes_sin'] = np.sin(2 * np.pi * (df['mes'] - 1) / 12)
    df['mes_cos'] = np.cos(2 * np.pi * (df['mes'] - 1) / 12)

    df['diff_prob'] = df['prob_mais'] - df['prob_ate']

    return df

def carregar_base_existente(caminho_arquivo):
    if not os.path.exists(caminho_arquivo):
        return pd.DataFrame()

    df_existente = pd.read_excel(caminho_arquivo)

    if 'id' in df_existente.columns:
        df_existente['id'] = df_existente['id'].astype(str)

    return df_existente

def buscar_novos_registros(ids_existentes, max_paginas=200, paginas_sem_novidade_limite=3):
    url_base = "https://app.palpita.io/api/v1/markets"
    limit_por_pagina = 100
    termo_busca = "Rodovia (5 minutos): quantos carros?"

    pagina = 1
    paginas_sem_novidade = 0
    lista_novos = []

    print("Buscando apenas registros novos...")

    while pagina <= max_paginas:
        print(f"Consultando página {pagina}...")

        params = {
            'page': pagina,
            'limit': limit_por_pagina,
            'status': 'RESOLVED',
            'search': termo_busca,
            'orderBy': 'closesAt',
            'orderDirection': 'DESC'
        }

        try:
            response = requests.get(url_base, params=params, timeout=15)
            response.raise_for_status()
            dados = response.json()
            items = dados.get('data', {}).get('items', [])

            if not items:
                print("Página vazia. Encerrando busca.")
                break

            qtd_novos_pagina = 0

            for item in items:
                item_id = str(item.get('id'))

                if item_id in ids_existentes:
                    continue

                descricao = item.get('description', '')
                dados_rodovia = extrair_dados_rodovia(descricao)

                if not dados_rodovia['rodovia_identificacao']:
                    continue

                selections = item.get('selections', [])
                res_id = item.get('resultSelectionId')

                if res_id is None:
                    print(f"[Ignorado] Mercado {item.get('id')} sem resultSelectionId")
                    continue

                vencedor = next((s.get('label') for s in selections if s.get('id') == res_id), None)

                if vencedor is None:
                    print(f"[Ignorado] Mercado {item.get('id')} sem seleção vencedora válida")
                    continue

                meta = next((s.get('label', '').split()[-1] for s in selections if s.get('label')), "0")

                lista_novos.append({
                    'id': item_id,
                    'rodovia_identificacao': dados_rodovia['rodovia_identificacao'],
                    'rodovia': dados_rodovia['rodovia'],
                    'km': dados_rodovia['km'],
                    'cidade': dados_rodovia['cidade'],
                    'uf': dados_rodovia['uf'],
                    'abertura': item.get('opensAt'),
                    'fechamento': item.get('closesAt'),
                    'resolvido_em': item.get('resolvedAt'),
                    'meta_referencia': meta,
                    'resultado_vencedor': vencedor,
                    'prob_mais': selections[0].get('impliedProb') if len(selections) > 0 else None,
                    'prob_ate': selections[1].get('impliedProb') if len(selections) > 1 else None,
                })

                qtd_novos_pagina += 1

            print(f"Novos encontrados nesta página: {qtd_novos_pagina}")

            if qtd_novos_pagina == 0:
                paginas_sem_novidade += 1
            else:
                paginas_sem_novidade = 0

            if paginas_sem_novidade >= paginas_sem_novidade_limite:
                print(f"Sem novidade por {paginas_sem_novidade_limite} páginas seguidas. Encerrando busca.")
                break

            pagina += 1
            time.sleep(0.4)

        except Exception as e:
            print(f"Erro na página {pagina}: {e}")
            break

    df_novos = pd.DataFrame(lista_novos)
    return df_novos

def atualizar_dados_rodovias(caminho_arquivo):
    pasta_arquivo = os.path.dirname(caminho_arquivo)
    if pasta_arquivo and not os.path.exists(pasta_arquivo):
        os.makedirs(pasta_arquivo, exist_ok=True)

    arquivo_existe = os.path.exists(caminho_arquivo)
    df_existente = carregar_base_existente(caminho_arquivo)

    if not arquivo_existe:
        print("Planilha não encontrada. Criando carga inicial...")
        ids_existentes = set()
    else:
        ids_existentes = set(df_existente['id'].astype(str).dropna().tolist()) if 'id' in df_existente.columns else set()
        print(f"Total de IDs já existentes: {len(ids_existentes)}")

    df_novos = buscar_novos_registros(ids_existentes=ids_existentes)

    if df_novos.empty:
        if not arquivo_existe:
            print("Nenhum registro encontrado para criar a planilha inicial.")
        else:
            print("Nenhum registro novo encontrado.")
        return

    df_novos = tratar_dataframe(df_novos)

    if arquivo_existe and not df_existente.empty:
        df_final = pd.concat([df_existente, df_novos], ignore_index=True)
    else:
        df_final = df_novos.copy()

    if 'id' in df_final.columns:
        df_final['id'] = df_final['id'].astype(str)
        df_final = df_final.drop_duplicates(subset=['id'], keep='first')

    if 'fechamento' in df_final.columns:
        df_final['fechamento'] = pd.to_datetime(df_final['fechamento'], errors='coerce')
        df_final = df_final.sort_values(by='fechamento', ascending=False)

    df_final.to_excel(caminho_arquivo, index=False)

    if not arquivo_existe:
        print(f"Planilha criada com sucesso: {caminho_arquivo}")
        print(f"Total inicial de registros: {len(df_final)}")
    else:
        print(f"Registros novos adicionados: {len(df_novos)}")
        print(f"Total final na planilha: {len(df_final)}")
        print(f"Planilha atualizada com sucesso: {caminho_arquivo}")



atualizar_dados_rodovias(ARQUIVO_BASE)