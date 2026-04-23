import requests
import pandas as pd
import time
import re
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DADOS_DIR = BASE_DIR.parent / "DadosRodovias"


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

def buscar_dados_todas_rodovias(total_casos):
    url_base = "https://app.palpita.io/api/v1/markets"
    limit_por_pagina = 100
    lista_final = []

    termo_busca = "Rodovia (5 minutos): quantos carros?"
    pagina = 1

    max_paginas_vazias_seguidas = 5
    paginas_vazias_seguidas = 0
    max_paginas = 1000

    print(f"Buscando mercados de todas as rodovias (Meta: {total_casos} registros salvos)...")

    while len(lista_final) < total_casos and pagina <= max_paginas:
        percentual = (len(lista_final) / total_casos) * 100
        print(f"Progresso: {percentual:.1f}% | Itens salvos: {len(lista_final)} | Consultando página: {pagina}")

        params = {
            'page': pagina,
            'limit': limit_por_pagina,
            'status': 'RESOLVED',
            'search': termo_busca,
            'orderBy': 'closesAt',
            'orderDirection': 'DESC'
        }

        try:
            response = requests.get(url_base, params=params, timeout=10)
            response.raise_for_status()
            dados = response.json()
            items = dados.get('data', {}).get('items', [])

            if not items:
                paginas_vazias_seguidas += 1
                print(f"[Aviso] Página {pagina} veio vazia. Tentativa {paginas_vazias_seguidas}/{max_paginas_vazias_seguidas}")

                if paginas_vazias_seguidas >= max_paginas_vazias_seguidas:
                    print("\n[Aviso] Muitas páginas vazias seguidas. Encerrando busca.")
                    break

                pagina += 1
                time.sleep(0.5)
                continue

            paginas_vazias_seguidas = 0

            for item in items:
                if len(lista_final) >= total_casos:
                    break

                descricao = item.get('description', '')
                dados_rodovia = extrair_dados_rodovia(descricao)

                if not dados_rodovia['rodovia_identificacao']:
                    continue

                selections = item.get('selections', [])
                res_id = item.get('resultSelectionId')

                # Se o mercado veio como resolvido, mas não trouxe o id da seleção vencedora, esse registro deve ser ignorado para não salvar resultado inconsistente.
                if res_id is None:
                    print(f"[Ignorado] Mercado {item.get('id')} sem resultSelectionId")
                    continue

                vencedor = next((s.get('label') for s in selections if s.get('id') == res_id), None)

                #se por algum motivo o resultSelectionId existir, mas não bater com nenhuma seleção, também ignora o registro.
                if vencedor is None:
                    print(f"[Ignorado] Mercado {item.get('id')} sem seleção vencedora válida")
                    continue
                

                meta = next((s.get('label', '').split()[-1] for s in selections if s.get('label')), "0")

                lista_final.append({
                    'id': item.get('id'),
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

            pagina += 1
            time.sleep(0.3)

        except Exception as e:
            print(f"Erro na página {pagina}: {e}")
            break

    df = pd.DataFrame(lista_final)

    if not df.empty:
        df['abertura'] = pd.to_datetime(df['abertura'], errors='coerce').dt.tz_localize(None)
        df['fechamento'] = pd.to_datetime(df['fechamento'], errors='coerce').dt.tz_localize(None)
        df['resolvido_em'] = pd.to_datetime(df['resolvido_em'], errors='coerce').dt.tz_localize(None)
        df['dia_semana_num'] = df['abertura'].dt.dayofweek  #0 = Monday, 1 = Tuesday, ..., 6 = Sunday
        df['dia_semana'] = df['abertura'].dt.day_name()
        df["km"] = pd.to_numeric(df["km"], errors="coerce")
        df["meta_referencia"] = pd.to_numeric(df["meta_referencia"], errors="coerce")
        df["prob_mais"] = pd.to_numeric(df["prob_mais"], errors="coerce")
        df["prob_ate"] = pd.to_numeric(df["prob_ate"], errors="coerce")
        df["diff_prob"] = df["prob_mais"] - df["prob_ate"]

        df['hora_abertura'] = df['abertura'].dt.hour
        df['mes'] = df['abertura'].dt.month
        df['fim_semana'] = df['dia_semana_num'].isin([5, 6]).astype(int)

        df['hora_sin'] = np.sin(2 * np.pi * df['hora_abertura'] / 24)
        df['hora_cos'] = np.cos(2 * np.pi * df['hora_abertura'] / 24)

        df['dia_semana_sin'] = np.sin(2 * np.pi * df['dia_semana_num'] / 7)
        df['dia_semana_cos'] = np.cos(2 * np.pi * df['dia_semana_num'] / 7)

        df['mes_sin'] = np.sin(2 * np.pi * (df['mes'] - 1) / 12)
        df['mes_cos'] = np.cos(2 * np.pi * (df['mes'] - 1) / 12)

    
    return df


df_campos = buscar_dados_todas_rodovias(10000)

DADOS_DIR.mkdir(parents=True, exist_ok=True)
nome_arquivo = DADOS_DIR / "dados_todas_rodovias.xlsx"
df_campos.to_excel(nome_arquivo, index=False)

print(f"\nArquivo Excel salvo com sucesso: {nome_arquivo}")
print(f"Total de linhas extraídas: {len(df_campos)}")