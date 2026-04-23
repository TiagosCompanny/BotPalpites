import requests
import pandas as pd
import time
import re
#Método para obter quais rodovias compoe no site


def buscar_rodovias_distintas(total_rodovias):
    url_base = "https://app.palpita.io/api/v1/markets"
    limit_por_pagina = 100
    lista_final = []
    chaves_unicas = set()

    termo_busca = "Rodovia (5 minutos): quantos carros?"
    pagina = 1

    max_paginas_vazias_seguidas = 5
    paginas_vazias_seguidas = 0
    max_paginas = 1000

    print(f"Buscando rodovias distintas (Meta: {total_rodovias} rodovias únicas)...")

    while len(lista_final) < total_rodovias and pagina <= max_paginas:
        percentual = (len(lista_final) / total_rodovias) * 100
        print(f"Progresso: {percentual:.1f}% | Rodovias únicas salvas: {len(lista_final)} | Consultando página: {pagina}")

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
                if len(lista_final) >= total_rodovias:
                    break

                descricao = item.get('description', '')
                if not descricao or not descricao.strip():
                    continue

                primeira_linha = descricao.splitlines()[0].strip()

                chave = re.sub(r'\s+', ' ', primeira_linha).strip().lower()

                if chave in chaves_unicas:
                    continue

                chaves_unicas.add(chave)

                match = re.match(r'^[•\-]?\s*(.*?),\s*KM\s*(\d+)\s*[—-]\s*(.*?)\s*\(([A-Z]{2})\)\.?$', primeira_linha)

                if match:
                    nome_rodovia = match.group(1).strip()
                    km = match.group(2).strip()
                    cidade = match.group(3).strip()
                    uf = match.group(4).strip()
                else:
                    nome_rodovia = None
                    km = None
                    cidade = None
                    uf = None

                lista_final.append({
                    'identificacao': primeira_linha,
                    'rodovia': nome_rodovia,
                    'km': km,
                    'cidade': cidade,
                    'uf': uf,
                    'descricao_completa_exemplo': descricao
                })

            pagina += 1
            time.sleep(0.3)

        except Exception as e:
            print(f"Erro na página {pagina}: {e}")
            break

    df = pd.DataFrame(lista_final)
    return df

df_rodovias = buscar_rodovias_distintas(1000)

nome_arquivo = "rodovias_distintas.xlsx"
df_rodovias.to_excel(nome_arquivo, index=False)

print(f"\nArquivo Excel salvo com sucesso: {nome_arquivo}")
print(f"Total de rodovias únicas extraídas: {len(df_rodovias)}")


#RETORNO:
"""
identificacao
• Rodovia Arão Sahm, KM 95 — Bragança Paulista (SP).
• Doutor Manoel Hyppolito Rego, KM 83 — Caraguatatuba (SP).
• Floriano Rodrigues Pinheiro, KM 46 — Campos do Jordão (SP).
• Doutor Manoel Hyppolito Rego, KM 110 — Caraguatatuba (SP).
• Floriano Rodrigues Pinheiro, KM 26 — Pindamonhangaba (SP).

"""