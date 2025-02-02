// scripts.js
document.addEventListener('DOMContentLoaded', function() {
    const form = document.getElementById('form-jogos');
    const resultadosDiv = document.getElementById('resultados');

    form.addEventListener('submit', async function(e) {
        e.preventDefault();
        
        // Mostrar loading
        resultadosDiv.innerHTML = `
            <div class="loading">
                <div class="spinner-border text-primary" role="status">
                    <span class="visually-hidden">Gerando jogos...</span>
                </div>
                <p class="mt-2">Gerando jogos...</p>
            </div>
        `;

        try {
            const quantidade = document.getElementById('quantidade_jogos').value;
            const estrategia = document.getElementById('estrategia').value;

            console.log('Enviando requisição:', {
                quantidade_jogos: parseInt(quantidade),
                estrategia: estrategia
            });

            const response = await fetch('/gerar', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    quantidade_jogos: parseInt(quantidade),
                    estrategia: estrategia
                })
            });

            console.log('Resposta do servidor:', response);
            const data = await response.json();
            console.log('Dados recebidos:', data);

            if (data.status === 'success') {
                renderizarJogos(data);

                // Adicionar funcionalidade de cópia
                document.querySelectorAll('.copy-button').forEach(button => {
                    button.addEventListener('click', function() {
                        const numbers = this.dataset.numbers;
                        navigator.clipboard.writeText(numbers).then(() => {
                            const originalText = this.textContent;
                            this.textContent = 'Copiado!';
                            this.classList.add('btn-success');
                            this.classList.remove('btn-outline-primary');
                            
                            setTimeout(() => {
                                this.textContent = originalText;
                                this.classList.remove('btn-success');
                                this.classList.add('btn-outline-primary');
                            }, 1500);
                        });
                    });
                });

            } else {
                resultadosDiv.innerHTML = `
                    <div class="error-message">
                        Erro ao gerar jogos: ${data.message}
                    </div>
                `;
            }
        } catch (error) {
            console.error('Erro detalhado:', error);
            resultadosDiv.innerHTML = `
                <div class="error-message">
                    Erro ao comunicar com o servidor: ${error.message}
                </div>
            `;
        }
    });
});

function renderizarJogos(data) {
    const resultadosDiv = document.getElementById('resultados');
    if (!resultadosDiv) {
        console.error('Elemento resultados não encontrado');
        return;
    }

    resultadosDiv.innerHTML = '';

    data.jogos.forEach((jogo, index) => {
        // Criar o card do jogo
        const jogoCard = document.createElement('div');
        jogoCard.className = 'card mb-3';

        // Gerar HTML para os números do jogo
        const numerosHtml = jogo.jogo.map(num => 
            `<span class="numero-bola">${num}</span>`
        ).join('');

        // Construir o HTML do card
        jogoCard.innerHTML = `
            <div class="card-header">
                <h5 class="card-title">Jogo ${index + 1}</h5>
            </div>
            <div class="card-body">
                <div class="numeros-jogo mb-3">
                    ${numerosHtml}
                    <button class="btn btn-copy" data-numbers="${jogo.jogo.join(',')}">
                        <i class="fas fa-copy"></i> Copiar números
                    </button>
                </div>
                
                <h6>Estatísticas de Acertos:</h6>
                <ul class="list-group mb-3">
                    <li class="list-group-item">11 números: ${jogo.estatisticas['11']} vezes</li>
                    <li class="list-group-item">12 números: ${jogo.estatisticas['12']} vezes</li>
                    <li class="list-group-item">13 números: ${jogo.estatisticas['13']} vezes</li>
                    <li class="list-group-item">14 números: ${jogo.estatisticas['14']} vezes</li>
                    <li class="list-group-item">15 números: ${jogo.estatisticas['15']} vezes</li>
                </ul>

                <h6>Melhores Coincidências com Concursos Anteriores:</h6>
                <div class="table-responsive">
                    <table class="table table-striped">
                        <thead>
                            <tr>
                                <th>Concurso</th>
                                <th>Data</th>
                                <th>Acertos</th>
                                <th>Números Coincidentes</th>
                            </tr>
                        </thead>
                        <tbody id="comparacoes-${index + 1}">
                        </tbody>
                    </table>
                </div>
            </div>
        `;

        // Adicionar o card ao container de resultados
        resultadosDiv.appendChild(jogoCard);

        // Adicionar as comparações à tabela
        const tabelaBody = jogoCard.querySelector(`#comparacoes-${index + 1}`);
        if (tabelaBody && jogo.comparacoes) {
            jogo.comparacoes.forEach(comp => {
                const tr = document.createElement('tr');
                tr.innerHTML = `
                    <td>${comp.concurso}</td>
                    <td>${comp.data_concurso}</td>
                    <td>${comp.numeros_acertados}</td>
                    <td>${comp.numeros_coincidentes.map(n => 
                        `<span class="numero-coincidente">${n}</span>`).join('')}</td>
                `;
                tabelaBody.appendChild(tr);
            });
        }

        // Adicionar funcionalidade de cópia
        const copyButton = jogoCard.querySelector('.btn-copy');
        copyButton.addEventListener('click', async function() {
            const numbers = this.dataset.numbers;
            try {
                await navigator.clipboard.writeText(numbers);
                this.classList.add('copied');
                this.innerHTML = '<i class="fas fa-check"></i> Copiado!';
                setTimeout(() => {
                    this.classList.remove('copied');
                    this.innerHTML = '<i class="fas fa-copy"></i> Copiar números';
                }, 2000);
            } catch (err) {
                console.error('Erro ao copiar:', err);
            }
        });
    });

    // Adicionar os gráficos
    criarGraficos(data);
}

// Variável global para armazenar as instâncias dos gráficos
let graficos = {
    acertos: null,
    comparacao: null,
    probabilidade: null
};

function criarGraficos(data) {
    // Destruir gráficos existentes
    Object.values(graficos).forEach(grafico => {
        if (grafico) {
            grafico.destroy();
        }
    });

    // Gráfico de Acertos
    const ctxAcertos = document.getElementById('graficoAcertos').getContext('2d');
    graficos.acertos = new Chart(ctxAcertos, {
        type: 'bar',
        data: {
            labels: ['11', '12', '13', '14', '15'],
            datasets: [{
                label: 'Quantidade de Acertos',
                data: [
                    data.jogos[0].estatisticas['11'],
                    data.jogos[0].estatisticas['12'],
                    data.jogos[0].estatisticas['13'],
                    data.jogos[0].estatisticas['14'],
                    data.jogos[0].estatisticas['15']
                ],
                backgroundColor: [
                    'rgba(99, 102, 241, 0.5)',
                    'rgba(139, 92, 246, 0.5)',
                    'rgba(16, 185, 129, 0.5)',
                    'rgba(59, 130, 246, 0.5)',
                    'rgba(217, 70, 239, 0.5)'
                ],
                borderColor: [
                    'rgba(99, 102, 241, 1)',
                    'rgba(139, 92, 246, 1)',
                    'rgba(16, 185, 129, 1)',
                    'rgba(59, 130, 246, 1)',
                    'rgba(217, 70, 239, 1)'
                ],
                borderWidth: 1
            }]
        },
        options: {
            responsive: true,
            plugins: {
                legend: {
                    position: 'top',
                },
                title: {
                    display: true,
                    text: 'Distribuição de Acertos'
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    ticks: {
                        stepSize: 1
                    }
                }
            }
        }
    });

    // Gráfico de Comparação
    const ctxComparacao = document.getElementById('graficoComparacao').getContext('2d');
    graficos.comparacao = new Chart(ctxComparacao, {
        type: 'line',
        data: {
            labels: data.jogos[0].comparacoes.map(c => `Concurso ${c.concurso}`),
            datasets: [{
                label: 'Números Acertados',
                data: data.jogos[0].comparacoes.map(c => c.numeros_acertados),
                borderColor: 'rgba(99, 102, 241, 1)',
                backgroundColor: 'rgba(99, 102, 241, 0.1)',
                tension: 0.4,
                fill: true
            }]
        },
        options: {
            responsive: true,
            plugins: {
                legend: {
                    position: 'top',
                },
                title: {
                    display: true,
                    text: 'Histórico de Acertos'
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    max: 15,
                    ticks: {
                        stepSize: 1
                    }
                }
            }
        }
    });

    // Gráfico de Probabilidade
    const ctxProbabilidade = document.getElementById('graficoProbabilidade').getContext('2d');
    const probabilidades = calcularProbabilidadesPorRegiao(data.jogos[0].jogo);
    graficos.probabilidade = new Chart(ctxProbabilidade, {
        type: 'doughnut',
        data: {
            labels: ['1-5', '6-10', '11-15', '16-20', '21-25'],
            datasets: [{
                data: probabilidades,
                backgroundColor: [
                    'rgba(99, 102, 241, 0.7)',
                    'rgba(139, 92, 246, 0.7)',
                    'rgba(16, 185, 129, 0.7)',
                    'rgba(59, 130, 246, 0.7)',
                    'rgba(217, 70, 239, 0.7)'
                ],
                borderWidth: 1
            }]
        },
        options: {
            responsive: true,
            plugins: {
                legend: {
                    position: 'top',
                },
                title: {
                    display: true,
                    text: 'Distribuição por Região'
                }
            }
        }
    });

    // Atualizar estatísticas
    const mediaAcertos = calcularMediaAcertos(data.jogos[0].estatisticas);
    document.getElementById('mediaAcertos').textContent = mediaAcertos.toFixed(1);
    document.getElementById('melhorResultado').textContent = 
        Math.max(...Object.values(data.jogos[0].estatisticas));
    document.getElementById('probabilidadeAcertos').textContent = 
        calcularProbabilidade(data.jogos[0].estatisticas) + '%';
}

function calcularMediaAcertos(estatisticas) {
    const total = Object.entries(estatisticas).reduce((acc, [acertos, qtd]) => {
        return acc + (parseInt(acertos) * qtd);
    }, 0);
    const quantidade = Object.values(estatisticas).reduce((a, b) => a + b, 0);
    return total / quantidade;
}

function calcularProbabilidade(estatisticas) {
    const total = Object.values(estatisticas).reduce((a, b) => a + b, 0);
    return ((total / 2583) * 100).toFixed(2); // 2583 é o total de jogos da Lotofácil
}

function calcularProbabilidadesPorRegiao(jogo) {
    const regioes = [0, 0, 0, 0, 0]; // Contadores para cada região
    jogo.forEach(num => {
        if (num <= 5) regioes[0]++;
        else if (num <= 10) regioes[1]++;
        else if (num <= 15) regioes[2]++;
        else if (num <= 20) regioes[3]++;
        else regioes[4]++;
    });
    return regioes.map(count => (count / jogo.length) * 100);
}