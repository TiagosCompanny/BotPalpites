import os
import requests
import pandas as pd
import time
import re
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
ARQUIVO_BASE = BASE_DIR.parent / "DadosPassagens" / "dados_todas_passagens.xlsx"

def extrair_dados_passagem(descricao):
    if not descricao or not descricao.strip():
        return {
            'local_identificacao': None,
            'local': None,
            'cidade': None
        }

    primeira_linha = descricao.splitlines()[0].strip()
    primeira_linha = primeira_linha.lstrip('•').strip().rstrip('.')

    padrao = r'^(.*?)\s*[—-]\s*(.*?)$'
    match = re.match(padrao, primeira_linha)

    if match:
        return {
            'local_identificacao': primeira_linha,
            'local': match.group(1).strip(),
            'cidade': match.group(2).strip()
        }

    return {
        'local_identificacao': primeira_linha,
        'local': primeira_linha,
        'cidade': None
    }

def tratar_dataframe(df):
    if df.empty:
        return df

    df['abertura'] = pd.to_datetime(df['abertura'], errors='coerce').dt.tz_localize(None)
    df['fechamento'] = pd.to_datetime(df['fechamento'], errors='coerce').dt.tz_localize(None)
    df['resolvido_em'] = pd.to_datetime(df['resolvido_em'], errors='coerce').dt.tz_localize(None)

    # 🔥 CORREÇÃO 2: TRATAMENTO DA META (Substitui vírgula por ponto antes de converter)
    if 'meta_referencia' in df.columns:
        df['meta_referencia'] = df['meta_referencia'].astype(str).str.replace(',', '.', regex=False)
        df['meta_referencia'] = pd.to_numeric(df['meta_referencia'], errors='coerce')

    df['dia_semana_num'] = df['abertura'].dt.dayofweek
    df['dia_semana'] = df['abertura'].dt.day_name()
    df['hora_abertura'] = df['abertura'].dt.hour
    df['minuto_abertura'] = df['abertura'].dt.minute
    df['mes'] = df['abertura'].dt.month
    df['fim_semana'] = df['dia_semana_num'].isin([5, 6]).astype(int)

    df['hora_sin'] = np.sin(2 * np.pi * df['hora_abertura'] / 24)
    df['hora_cos'] = np.cos(2 * np.pi * df['hora_abertura'] / 24)

    df['dia_semana_sin'] = np.sin(2 * np.pi * df['dia_semana_num'] / 7)
    df['dia_semana_cos'] = np.cos(2 * np.pi * df['dia_semana_num'] / 7)

    df['mes_sin'] = np.sin(2 * np.pi * (df['mes'] - 1) / 12)
    df['mes_cos'] = np.cos(2 * np.pi * (df['mes'] - 1) / 12)

    return df

def carregar_base_existente(caminho_arquivo):
    if not os.path.exists(caminho_arquivo):
        return pd.DataFrame()

    df_existente = pd.read_excel(caminho_arquivo)

    if 'id' in df_existente.columns:
        # 🔥 CORREÇÃO 3: LIMPEZA DE IDs (Remove o ".0" fantasma do Excel)
        df_existente['id'] = df_existente['id'].astype(str).str.replace(r'\.0$', '', regex=True)

    return df_existente

def buscar_novos_registros(ids_existentes, max_paginas=200, paginas_sem_novidade_limite=3):
    url_base = "https://app.palpita.io/api/v1/markets"
    limit_por_pagina = 100
    termo_busca = "Rua (4m 40s): quantas passagens?"

    pagina = 1
    paginas_sem_novidade = 0
    lista_novos = []

    print("Buscando apenas registros novos de passagens...")

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
                dados_passagem = extrair_dados_passagem(descricao)

                if not dados_passagem['local_identificacao']:
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

                meta = item.get('valueNeeded')

                if meta is None:
                    meta = next((s.get('label', '').split()[-1] for s in selections if s.get('label')), None)

                lista_novos.append({
                    'id': item_id,
                    'mercado_titulo': item.get('title'),
                    'local_identificacao': dados_passagem['local_identificacao'],
                    'local': dados_passagem['local'],
                    'cidade': dados_passagem['cidade'],
                    'tag': item.get('tag'),
                    'abertura': item.get('opensAt'),
                    'fechamento': item.get('closesAt'),
                    'resolvido_em': item.get('resolvedAt'),
                    'meta_referencia': meta,
                    'resultado_vencedor': vencedor,
                    'market_slug': item.get('slug'),
                    'market_status': item.get('status'),
                    'matching_system': item.get('matchingSystem'),
                    'selection_mais': selections[0].get('label') if len(selections) > 0 else None,
                    'selection_ate': selections[1].get('label') if len(selections) > 1 else None,
                    'selection_mais_id': selections[0].get('id') if len(selections) > 0 else None,
                    'selection_ate_id': selections[1].get('id') if len(selections) > 1 else None
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

def atualizar_dados_passagens(caminho_arquivo):
    caminho_arquivo = Path(caminho_arquivo)
    caminho_arquivo.parent.mkdir(parents=True, exist_ok=True)

    df_existente = carregar_base_existente(caminho_arquivo)

    if df_existente.empty:
        print("Planilha não encontrada ou vazia. Criando base inicial...")

        df_inicial = buscar_novos_registros(
            ids_existentes=set(),
            max_paginas=1000,
            paginas_sem_novidade_limite=10
        )

        if df_inicial.empty:
            print("Nenhum registro encontrado para criar a base inicial.")
            return

        df_inicial = tratar_dataframe(df_inicial)
        
        # 🔥 CORREÇÃO 3: Limpeza de IDs antes do drop_duplicates
        df_inicial['id'] = df_inicial['id'].astype(str).str.replace(r'\.0$', '', regex=True)
        df_inicial = df_inicial.drop_duplicates(subset=['id'], keep='first')

        if 'fechamento' in df_inicial.columns:
            df_inicial['fechamento'] = pd.to_datetime(df_inicial['fechamento'], errors='coerce')
            df_inicial = df_inicial.sort_values(by='fechamento', ascending=False)

        df_inicial.to_excel(caminho_arquivo, index=False)

        print(f"Base inicial criada com sucesso: {caminho_arquivo}")
        print(f"Total inicial na planilha: {len(df_inicial)}")
        return

    ids_existentes = set(df_existente['id'].dropna().tolist())
    print(f"Total de IDs já existentes: {len(ids_existentes)}")

    df_novos = buscar_novos_registros(ids_existentes=ids_existentes)

    if df_novos.empty:
        print("Nenhum registro novo encontrado.")
        return

    df_novos = tratar_dataframe(df_novos)

    df_final = pd.concat([df_existente, df_novos], ignore_index=True)
    
    # 🔥 CORREÇÃO 3: Limpeza de IDs na base final antes do drop_duplicates
    df_final['id'] = df_final['id'].astype(str).str.replace(r'\.0$', '', regex=True)
    df_final = df_final.drop_duplicates(subset=['id'], keep='first')

    if 'fechamento' in df_final.columns:
        df_final['fechamento'] = pd.to_datetime(df_final['fechamento'], errors='coerce')
        df_final = df_final.sort_values(by='fechamento', ascending=False)

    df_final.to_excel(caminho_arquivo, index=False)

    print(f"Registros novos adicionados: {len(df_novos)}")
    print(f"Total final na planilha: {len(df_final)}")
    print(f"Planilha atualizada com sucesso: {caminho_arquivo}")

atualizar_dados_passagens(ARQUIVO_BASE)