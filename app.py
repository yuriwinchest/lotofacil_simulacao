import matplotlib
matplotlib.use('Agg')
from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import os
from typing import List, Dict, Any
import logging
from threading import Lock
from datetime import datetime
from functools import lru_cache, wraps
import json
# Novas bibliotecas para machine learning
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import seaborn as sns # Adicionando o seaborn
import requests
from services.loteria_api import LoteriaAPI
# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Lock para sincronizar acesso ao matplotlib
plot_lock = Lock()

app = Flask(__name__)

# Constantes
ARQUIVO_DADOS = "Lotofácil-atualizado.xlsx"
SHEET_NAME = "LOTOFÁCIL"
TOTAL_NUMEROS = 25
NUMEROS_POR_JOGO = 15

# Instancia o cliente da API
loteria_api = LoteriaAPI()

def validate_request_data(f):
    """Decorator para validar dados do request."""
    @wraps(f)
    def wrapper(*args, **kwargs):
        if not request.is_json:
            return jsonify({"status": "error", "message": "Requisição deve ser JSON"}), 400
        data = request.get_json()
        if not data:
            return jsonify({"status": "error", "message": "Dados não fornecidos"}), 400
        return f(*args, **kwargs, data=data)
    return wrapper
    
def feature_engineering(df):
    """Função para engenharia de atributos"""
    for i in range(1, TOTAL_NUMEROS + 1):
        df[f'numero_{i}'] = df.apply(lambda row: 1 if i in row.values else 0, axis=1)
    df['soma_numeros'] = df[[f'numero_{i}' for i in range(1, TOTAL_NUMEROS + 1)]].sum(axis=1)
    return df

class LotofacilAnalyzer:
    def __init__(self):
        self.df = None
        self.frequencias = None
        plt.ioff()
        self._last_update = None
        self.cached_padroes = None
        self.load_data()

    def load_data(self):
        try:
            # Define os possíveis caminhos dos arquivos na pasta data
            arquivo_novo = os.path.join('data', 'lotofacil.xlsx')
            arquivo_antigo = os.path.join('data', 'Lotofácil-atualizado.xlsx')
            
            # Tenta carregar primeiro o arquivo novo, depois o antigo
            if os.path.exists(arquivo_novo):
                self.df = pd.read_excel(arquivo_novo)
                logger.info(f"Dados carregados do arquivo: {arquivo_novo}")
            elif os.path.exists(arquivo_antigo):
                self.df = pd.read_excel(arquivo_antigo)
                logger.info(f"Dados carregados do arquivo: {arquivo_antigo}")
            else:
                # Se nenhum arquivo for encontrado, baixa os dados da API
                logger.info("Arquivos não encontrados. Baixando dados da API...")
                from services.loteria_api import LoteriaAPI
                api = LoteriaAPI()
                resultados = api.get_last_contests('lotofacil', 100)
                
                # Converte para DataFrame
                dados = []
                for res in resultados:
                    row = {
                        'Concurso': res['concurso'],
                        'Data Sorteio': res['data']
                    }
                    for i, num in enumerate(res['numeros'], 1):
                        row[f'Bola{i}'] = num
                    row['Rateio 15 acertos'] = res['premio']
                    row['Ganhadores 15 acertos'] = res['ganhadores']
                    dados.append(row)
                
                self.df = pd.DataFrame(dados)
                
                # Salva os dados baixados
                os.makedirs('data', exist_ok=True)
                self.df.to_excel(arquivo_novo, index=False)
                logger.info(f"Novos dados salvos em: {arquivo_novo}")
            
        except Exception as e:
            logger.error(f"Erro ao carregar dados: {e}")
            # Cria um DataFrame vazio em caso de erro
            self.df = pd.DataFrame()

    @lru_cache(maxsize=1)
    def calcular_frequencias(self) -> pd.DataFrame:
        """Calcula a frequência dos números sorteados com cache."""
        try:
            if self.df is None:
                self.load_data()

            todos_numeros = self.df.loc[:, "Bola1":"Bola15"].values.flatten()
            frequencia = pd.Series(todos_numeros).value_counts().reset_index()
            frequencia.columns = ["Número", "Frequência"]
            total_sorteios = len(self.df)
            frequencia['Probabilidade'] = (frequencia['Frequência'] / total_sorteios * 100).round(2)

            self.frequencias = frequencia.sort_values(by="Frequência", ascending=False)
            return self.frequencias

        except Exception as e:
            logger.error(f"Erro ao calcular frequências: {e}", exc_info=True)
            return pd.DataFrame(columns=["Número", "Frequência", "Probabilidade"])

    def gerar_grafico_frequencia(self) -> str:
        """Gera gráfico de frequência dos números de forma thread-safe."""
        with plot_lock:
            try:
                if self.frequencias is None:
                    self.calcular_frequencias()

                plt.clf()
                fig = plt.figure(figsize=(12, 6))
                bars = plt.bar(
                    self.frequencias['Número'],
                    self.frequencias['Frequência'],
                    color='#2563eb',
                    alpha=0.7
                )

                for bar in bars:
                    height = bar.get_height()
                    plt.text(
                        bar.get_x() + bar.get_width() / 2.,
                        height,
                        f'{int(height)}',
                        ha='center',
                        va='bottom'
                    )

                plt.xlabel('Número')
                plt.ylabel('Frequência')
                plt.title('Frequência dos Números na Lotofácil')
                plt.xticks(range(1, TOTAL_NUMEROS + 1))
                plt.grid(axis='y', linestyle='--', alpha=0.7)
                plt.tight_layout()

                os.makedirs('static', exist_ok=True)
                filepath = os.path.join('static', 'frequencia.png')
                fig.savefig(filepath, dpi=300, bbox_inches='tight')
                plt.close(fig)

                logger.info("Gráfico gerado com sucesso")
                return filepath

            except Exception as e:
                logger.error(f"Erro ao gerar gráfico: {e}", exc_info=True)
                plt.close('all')
                raise

    def gerar_jogos(self, quantidade: int, estrategia: str = 'misturados') -> List[List[int]]:
        """Gera jogos com estratégias melhoradas baseadas em análise de padrões."""
        try:
            if self.frequencias is None:
                self.calcular_frequencias()

            # Busca padrões no cache ou calcula se não existir
            if self.cached_padroes is None:
                self.cached_padroes = self.calcular_padroes_vencedores()
            padroes = self.cached_padroes

            jogos = []
            for _ in range(quantidade):
                jogo_valido = False
                tentativas = 0
                while not jogo_valido and tentativas < 200:  # Limitando as tentativas por segurança
                    if estrategia == 'frequentes':
                        jogo = self._gerar_jogo_frequentes()
                    elif estrategia == 'misturados':
                        jogo = self._gerar_jogo_misturado(padroes)
                    elif estrategia == 'balanceado':
                        jogo = self._gerar_jogo_balanceado(padroes)
                    else:
                        jogo = self._gerar_jogo_aleatorio(padroes)
                    if self._validar_jogo(jogo, padroes):
                        jogo_valido = True
                        jogos.append(sorted(jogo))
                    tentativas += 1

                if not jogo_valido:
                    logger.warning(f"Não foi possível gerar um jogo válido após {tentativas} tentativas para a estratégia {estrategia}. Gerando um jogo aleatório.")
                    jogo = self._gerar_jogo_aleatorio(padroes)
                    jogos.append(sorted(jogo))

            return jogos

        except Exception as e:
            logger.error(f"Erro ao gerar jogos: {e}", exc_info=True)
            raise

    def _gerar_jogo_frequentes(self) -> List[int]:
      """Gera jogo usando apenas números mais frequentes."""
      numeros_frequentes = self.frequencias['Número'].tolist()[:18]  # Pega os 18 números mais frequentes
      return random.sample(numeros_frequentes, NUMEROS_POR_JOGO)

    def _gerar_jogo_balanceado(self, padroes: Dict[str, Any]) -> List[int]:
        """Gera jogo balanceado baseado nos padrões históricos."""
        try:
            # Obtém a média ideal de pares e ímpares
            pares_ideal = int(round(padroes.get('pares_impares_media', [7, 8])[0]))
            impares_ideal = NUMEROS_POR_JOGO - pares_ideal

            # Separa números pares e ímpares
            numeros_pares = [n for n in range(1, TOTAL_NUMEROS + 1) if n % 2 == 0]
            numeros_impares = [n for n in range(1, TOTAL_NUMEROS + 1) if n % 2 != 0]

            # Gera o jogo com a distribuição ideal
            jogo = random.sample(numeros_pares, pares_ideal)
            jogo.extend(random.sample(numeros_impares, impares_ideal))

            return jogo
        except Exception as e:
            logger.error(f"Erro ao gerar jogo balanceado: {e}", exc_info=True)
            return random.sample(range(1, TOTAL_NUMEROS + 1), NUMEROS_POR_JOGO)

    def _gerar_jogo_aleatorio(self, padroes: Dict[str, Any]) -> List[int]:
        """Gera jogo aleatório considerando alguns padrões básicos."""
        try:
            # Tenta gerar um jogo que respeite a soma média
            soma_media = padroes.get('soma_total_media', 180)
            tentativas = 0
            while tentativas < 100:  # Limite de tentativas para evitar loop infinito
                jogo = random.sample(range(1, TOTAL_NUMEROS + 1), NUMEROS_POR_JOGO)
                if abs(sum(jogo) - soma_media) <= 20:  # Tolerância de ±20 na soma
                    return jogo
                tentativas += 1

            # Se não conseguir após as tentativas, retorna um jogo totalmente aleatório
            return random.sample(range(1, TOTAL_NUMEROS + 1), NUMEROS_POR_JOGO)
        except Exception as e:
            logger.error(f"Erro ao gerar jogo aleatório: {e}", exc_info=True)
            return random.sample(range(1, TOTAL_NUMEROS + 1), NUMEROS_POR_JOGO)

    def _gerar_jogo_misturado(self, padroes: Dict[str, Any]) -> List[int]:
      """Gera jogo misturando números frequentes e raros com base nos padrões."""
      try:
        numeros_frequentes = self.frequencias['Número'].tolist()[:15]
        numeros_menos_frequentes = self.frequencias['Número'].tolist()[15:]

        # Ajusta a quantidade de números frequentes baseado nos padrões
        qtd_frequentes = int(round(10 * (padroes.get('soma_total_media', 180) / 180)))
        qtd_frequentes = max(7, min(12, qtd_frequentes))  # Garante entre 7 e 12 números frequentes

        jogo = random.sample(numeros_frequentes, qtd_frequentes)
        jogo.extend(random.sample(numeros_menos_frequentes, NUMEROS_POR_JOGO - qtd_frequentes))

        return jogo
      except Exception as e:
          logger.error(f"Erro ao gerar jogo misturado: {e}", exc_info=True)
          return random.sample(range(1, TOTAL_NUMEROS + 1), NUMEROS_POR_JOGO)

    def _validar_jogo(self, jogo: List[int], padroes: Dict[str, Any]) -> bool:
        """Valida se o jogo gerado segue os padrões identificados."""
        try:
            # Conta pares e ímpares
            pares = len([n for n in jogo if n % 2 == 0])
            impares = 15 - pares
            
            # Calcula soma total
            soma = sum(jogo)
            
            # Verifica sequências
            jogo_ordenado = sorted(jogo)
            sequencias = sum(1 for i in range(len(jogo_ordenado)-1)
                            if jogo_ordenado[i+1] - jogo_ordenado[i] == 1)
            
            # Validações
            pares_ok = abs(pares - padroes.get('pares_impares_media', [7,8])[0]) <= 2
            soma_ok = abs(soma - padroes.get('soma_total_media', 180)) <= 20
            sequencias_ok = abs(sequencias - padroes.get('sequencias_media', 3)) <= 2
            
            return pares_ok and soma_ok and sequencias_ok
        except Exception as e:
            logger.error(f"Erro ao validar o jogo: {e}", exc_info=True)
            return False

    def avaliar_jogo(self, jogo: List[int]) -> Dict[str, Any]:
        """Avalia um jogo contra o histórico de resultados e mostra comparações detalhadas."""
        try:
            acertos = {"11": 0, "12": 0, "13": 0, "14": 0, "15": 0}
            comparacoes_detalhadas = []

            for idx, resultado in self.df.iterrows():
                numeros_sorteados = set(resultado['Bola1':'Bola15'])
                acertos_jogo = len(set(jogo).intersection(numeros_sorteados))

                if acertos_jogo >= 11:
                    acertos[str(acertos_jogo)] += 1
                    # Adiciona detalhes do concurso onde houve coincidência
                    concurso_info = {
                        'concurso': idx + 1,  # Assumindo que o índice começa em 0
                        'numeros_acertados': acertos_jogo,
                        'data_concurso': resultado.get('Data', 'Data não disponível'),
                        'numeros_coincidentes': list(set(jogo).intersection(numeros_sorteados))
                    }
                    comparacoes_detalhadas.append(concurso_info)
          
            # Ordena as comparações pelo número de acertos (decrescente)
            comparacoes_detalhadas.sort(key=lambda x: x['numeros_acertados'], reverse=True)
            
            return {
                "jogo": jogo,
                "estatisticas": acertos,
                "pontuacao": sum(int(k) * v for k, v in acertos.items()),
                "comparacoes": comparacoes_detalhadas[:5]  # Retorna os 5 melhores resultados
            }

        except Exception as e:
            logger.error(f"Erro ao avaliar jogo: {e}", exc_info=True)
            return {}

    def analisar_tendencias(self) -> Dict[str, Any]:
      """Analisa tendências nos últimos sorteios."""
      try:
        ultimos_jogos = self.df.head(10)  # Análise dos últimos 10 jogos
        numeros_recentes = ultimos_jogos.loc[:, "Bola1":"Bola15"].values.flatten()

        tendencias = {
            'numeros_quentes': pd.Series(numeros_recentes).value_counts().head(5).to_dict(),
            'numeros_ausentes': [n for n in range(1, TOTAL_NUMEROS + 1)
                              if n not in numeros_recentes],
            'pares_impares': {
                'pares': len([n for n in numeros_recentes if n % 2 == 0]),
                'impares': len([n for n in numeros_recentes if n % 2 != 0])
            }
        }
        return tendencias
      except Exception as e:
            logger.error(f"Erro ao analisar tendências: {e}", exc_info=True)
            return {}

    @lru_cache(maxsize=1)
    def calcular_padroes_vencedores(self) -> Dict[str, Any]:
      """Analisa padrões dos jogos vencedores."""
      try:
        jogos_vencedores = self.df.loc[:, "Bola1":"Bola15"].values
        padroes = {
            'pares_impares': [],
            'soma_total': [],
            'sequencias': [],
            'quadrantes': []
        }

        for jogo in jogos_vencedores:
            # Análise de pares e ímpares
            pares = len([n for n in jogo if n % 2 == 0])
            impares = 15 - pares
            padroes['pares_impares'].append((pares, impares))

            # Soma total dos números
            padroes['soma_total'].append(sum(jogo))

            # Análise de sequências
            jogo_ordenado = sorted(jogo)
            sequencias = 0
            for i in range(len(jogo_ordenado)-1):
                if jogo_ordenado[i+1] - jogo_ordenado[i] == 1:
                    sequencias += 1
            padroes['sequencias'].append(sequencias)

            # Análise por quadrantes
            quadrantes = [0] * 4
            for num in jogo:
                if num <= 6:
                    quadrantes[0] += 1
                elif num <= 12:
                    quadrantes[1] += 1
                elif num <= 18:
                    quadrantes[2] += 1
                else:
                    quadrantes[3] += 1
            padroes['quadrantes'].append(quadrantes)

        # Convertendo arrays NumPy para listas Python e calculando as médias
        return {
            'pares_impares_media': [float(x) for x in np.mean(padroes['pares_impares'], axis=0)],
            'soma_total_media': float(np.mean(padroes['soma_total'])),
            'sequencias_media': float(np.mean(padroes['sequencias'])),
            'quadrantes_media': [float(x) for x in np.mean(padroes['quadrantes'], axis=0)]
        }
      except Exception as e:
          logger.error(f"Erro ao calcular padrões: {e}", exc_info=True)
          return {}

# Instância global do analisador
analyzer = LotofacilAnalyzer()

@app.route('/')
def index():
    """Rota principal com análises melhoradas."""
    try:
        frequencias = analyzer.calcular_frequencias()
        analyzer.gerar_grafico_frequencia()
        tendencias = analyzer.analisar_tendencias()

        # Verifica as colunas disponíveis no DataFrame
        colunas_data = [col for col in analyzer.df.columns if 'data' in col.lower()]

        if colunas_data:
            ultima_atualizacao = analyzer.df[colunas_data[0]].max()
        else:
            ultima_atualizacao = "Data não disponível"

        estatisticas = {
            'total_jogos': len(analyzer.df),
            'ultima_atualizacao': ultima_atualizacao,
            'tendencias': tendencias
        }

        return render_template(
            'index.html',
            frequencias=frequencias.to_dict(orient='records'),
            estatisticas=estatisticas,
            erro=None
        )
    except Exception as e:
        logger.error(f"Erro na rota principal: {e}", exc_info=True)
        return render_template(
            'index.html',
            frequencias=[],
            estatisticas={
                'total_jogos': 0,
                'ultima_atualizacao': "Não disponível",
                'tendencias': {}
            },
            erro=f"Ocorreu um erro ao carregar os dados. Por favor, verifique se o arquivo Excel está correto."
        )

@app.route('/gerar', methods=['POST'])
@validate_request_data
def gerar_jogos(data=None):
    """Rota para geração de jogos com validação de entrada."""
    try:
        quantidade_jogos = data.get('quantidade_jogos')
        estrategia = data.get('estrategia', 'misturados')

        if not quantidade_jogos:
            return jsonify({"status": "error", "message": "Quantidade de jogos não especificada"}), 400

        try:
            quantidade_jogos = int(quantidade_jogos)
        except ValueError:
           return jsonify({"status": "error", "message": "Quantidade de jogos deve ser um número inteiro."}), 400

        if quantidade_jogos < 1 or quantidade_jogos > 100:
            return jsonify({"status": "error", "message": "Quantidade de jogos deve estar entre 1 e 100"}), 400

        if estrategia not in ['frequentes', 'misturados', 'balanceado', 'aleatorio']:
            return jsonify({"status": "error", "message": "Estratégia inválida."}), 400

        # Garante que os dados estão carregados
        if analyzer.df is None:
            analyzer.load_data()

        # Gera jogos com a nova estratégia
        jogos = analyzer.gerar_jogos(quantidade_jogos, estrategia)
        jogos_avaliados = [analyzer.avaliar_jogo(jogo) for jogo in jogos]

        # Adiciona informações sobre os padrões utilizados
        padroes = analyzer.calcular_padroes_vencedores()

        # Garante que todos os valores são serializáveis
        response_data = {
            "status": "success",
            "jogos": jogos_avaliados,
            "metadata": {
                "estrategia": estrategia,
                "quantidade": quantidade_jogos,
                "padroes_utilizados": {
                    k: v if isinstance(v, (int, float, str, list)) else float(v)
                    for k, v in padroes.items()
                },
                "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
        }
        return jsonify(response_data), 200

    except Exception as e:
        logger.error(f"Erro ao gerar jogos: {e}", exc_info=True)
        return jsonify({
            "status": "error",
            "message": f"Erro interno do servidor: {str(e)}"
        }), 500

@app.route('/ultimos-resultados')
def ultimos_resultados():
    """Retorna os últimos resultados da Lotofácil."""
    try:
        # Pega o último resultado do DataFrame
        ultimo_jogo = analyzer.df.iloc[0]
        
        # Extrai os números sorteados
        numeros = [int(ultimo_jogo[f'Bola{i}']) for i in range(1, 16)]
        
        # Formata a data (usando o nome correto da coluna 'Data Sorteio')
        data = ultimo_jogo['Data Sorteio'].strftime('%d/%m/%Y') if 'Data Sorteio' in ultimo_jogo else 'Data não disponível'
        
        # Pega o valor do prêmio (usando 'Rateio 15 acertos')
        premio = float(ultimo_jogo['Rateio 15 acertos']) if 'Rateio 15 acertos' in ultimo_jogo else 0.0
        
        return jsonify({
            'status': 'success',
            'resultado': {
                'concurso': int(ultimo_jogo['Concurso']),
                'data': data,
                'numeros': sorted(numeros),
                'premio': premio,
                'ganhadores': int(ultimo_jogo['Ganhadores 15 acertos'])
            }
        })
    except Exception as e:
        logger.error(f"Erro ao buscar últimos resultados: {e}", exc_info=True)
        return jsonify({
            'status': 'error',
            'message': 'Erro ao buscar resultados'
        }), 500

@app.route('/<loteria>/ultimos-resultados')
def resultados_loteria(loteria):
    """Retorna os resultados de uma loteria específica."""
    try:
        # Por enquanto, só temos dados da Lotofácil
        if loteria != 'lotofacil':
            return jsonify({
                'status': 'error',
                'message': f'Dados não disponíveis para {loteria}'
            }), 404

        # Usa os dados locais do Excel
        df_ordenado = analyzer.df.sort_values('Concurso', ascending=False)
        resultados = []
        
        for _, jogo in df_ordenado.head(10).iterrows():
            try:
                numeros = [int(jogo[f'Bola{i}']) for i in range(1, 16)]
                data = str(jogo['Data Sorteio'])
                premio = str(jogo['Rateio 15 acertos'])
                if isinstance(premio, str):
                    premio = float(premio.replace('R$', '').replace('.', '').replace(',', '.'))
                
                resultado = {
                    'concurso': int(jogo['Concurso']),
                    'data': data,
                    'numeros': sorted(numeros),
                    'premio': premio,
                    'ganhadores': int(jogo['Ganhadores 15 acertos'])
                }
                resultados.append(resultado)
                
            except Exception as e:
                logger.error(f"Erro ao processar jogo: {e}")
                continue
        
        if not resultados:
            return jsonify({
                'status': 'error',
                'message': 'Nenhum resultado encontrado'
            }), 404
        
        return jsonify({
            'status': 'success',
            'resultado': resultados[0],  # Último resultado
            'resultados': resultados     # Lista dos últimos resultados
        })
        
    except Exception as e:
        logger.error(f"Erro ao buscar resultados da {loteria}: {e}", exc_info=True)
        return jsonify({
            'status': 'error',
            'message': f'Erro ao buscar resultados da {loteria}'
        }), 500

@app.route('/todos-resultados')
def todos_resultados():
    """Retorna todos os resultados disponíveis."""
    try:
        # Usa os dados locais do Excel
        df_ordenado = analyzer.df.sort_values('Concurso', ascending=False)
        resultados = []
        
        for _, jogo in df_ordenado.head(50).iterrows():
            try:
                numeros = [int(jogo[f'Bola{i}']) for i in range(1, 16)]
                data = str(jogo['Data Sorteio'])
                premio = str(jogo['Rateio 15 acertos'])
                if isinstance(premio, str):
                    premio = float(premio.replace('R$', '').replace('.', '').replace(',', '.'))
                
                resultados.append({
                    'concurso': int(jogo['Concurso']),
                    'data': data,
                    'numeros': sorted(numeros),
                    'premio': premio,
                    'ganhadores': int(jogo['Ganhadores 15 acertos'])
                })
            except Exception as e:
                logger.error(f"Erro ao processar jogo: {e}")
                continue
        
        return jsonify({
            'status': 'success',
            'resultados': {'lotofacil': resultados}
        })
    except Exception as e:
        logger.error(f"Erro ao buscar todos os resultados: {e}", exc_info=True)
        return jsonify({
            'status': 'error',
            'message': 'Erro ao buscar resultados'
        }), 500

def limpar_valor_monetario(valor: str) -> float:
    """Converte string de valor monetário (R$ 1.234,56) para float."""
    try:
        if isinstance(valor, float):
            return valor
        if isinstance(valor, str):
            # Remove R$, pontos e substitui vírgula por ponto
            valor_limpo = valor.replace('R$', '').replace('.', '').replace(',', '.').strip()
            return float(valor_limpo) if valor_limpo != '' else 0.0
        return 0.0
    except Exception:
        return 0.0

if __name__ == '__main__':
    plt.ioff()
    app.run(debug=True)