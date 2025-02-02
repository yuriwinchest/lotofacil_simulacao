import requests
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class LoteriaAPI:
    def __init__(self):
        self.base_url = "https://loteriascaixa-api.herokuapp.com/api"
        self.endpoints = {
            'lotofacil': '/lotofacil/latest',
            'megasena': '/mega-sena/latest',
            'quina': '/quina/latest'
        }

    def get_latest_results(self, loteria):
        try:
            if loteria not in self.endpoints:
                raise ValueError(f"Loteria não suportada: {loteria}")

            url = f"{self.base_url}{self.endpoints[loteria]}"
            response = requests.get(url, timeout=10)
            response.raise_for_status()

            data = response.json()
            return self._format_result(data, loteria)
        except Exception as e:
            logger.error(f"Erro ao obter resultado de {loteria}: {e}")
            return None

    def get_last_contests(self, loteria, num_results=10):
        try:
            if loteria not in self.endpoints:
                raise ValueError(f"Loteria não suportada: {loteria}")

            url = f"{self.base_url}/{loteria}/latest/{num_results}"
            response = requests.get(url, timeout=10)
            response.raise_for_status()

            data = response.json()
            return [self._format_result(result, loteria) for result in data]
        except Exception as e:
            logger.error(f"Erro ao obter resultados de {loteria}: {e}")
            return []

    def _format_result(self, data, loteria):
        try:
            if loteria == 'lotofacil':
                return {
                    'concurso': int(data['concurso']),
                    'data': datetime.strptime(data['data'], '%d/%m/%Y').strftime('%d/%m/%Y'),
                    'numeros': sorted([int(n) for n in data['dezenas']]),
                    'premio': float(data['premiacoes'][0]['premio'].replace('R$', '').replace('.', '').replace(',', '.')),
                    'ganhadores': int(data['premiacoes'][0]['quantidade'])
                }
            elif loteria in ['megasena', 'quina']:
                return {
                    'concurso': int(data['concurso']),
                    'data': datetime.strptime(data['data'], '%d/%m/%Y').strftime('%d/%m/%Y'),
                    'numeros': sorted([int(n) for n in data['dezenas']]),
                    'premio': float(data['premiacoes'][0]['premio'].replace('R$', '').replace('.', '').replace(',', '.')),
                    'ganhadores': int(data['premiacoes'][0]['quantidade'])
                }
        except Exception as e:
            logger.error(f"Erro ao formatar resultado: {e}")
            return None
