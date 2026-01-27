// Script otimizado para extrair odds do painel Group do BetInAsia
// Retorna dados formatados para PAD (separados por quebra de linha)
// Use este script na ação "Executar função JavaScript" do Power Automate Desktop

function extrairOdds() {
    // Array para armazenar resultado
    let dados = [];
    
    try {
        // Buscar o painel Group pelo conteúdo característico
        const body = document.body.innerText;
        
        // Verificar se existe tabela de odds aberta
        if (!body.includes('Todos Os Agentes De Apostas') && !body.includes('MELHOR')) {
            return "ERRO: Tabela de odds não encontrada. Abra uma aposta primeiro.";
        }
        
        // Buscar todos os elementos do painel de apostas
        // O painel geralmente contém as classes do seletor: _62e8462d ou _4e93ff49
        let painel = document.querySelector('[class*="_62e8462d"]') ||
                     document.querySelector('[class*="_4e93ff49"]') ||
                     document.querySelector('[class*="_5f82213e"]');
        
        // Fallback: buscar pelo container que tem "Todos Os Agentes"
        if (!painel) {
            const divs = document.querySelectorAll('div');
            for (let div of divs) {
                if (div.innerText && 
                    div.innerText.includes('Todos Os Agentes De Apostas') &&
                    div.innerText.includes('MELHOR')) {
                    painel = div;
                    break;
                }
            }
        }
        
        if (!painel) {
            return "ERRO: Painel de apostas não localizado";
        }
        
        // Extrair todo o texto do painel
        const textoCompleto = painel.innerText;
        const linhas = textoCompleto.split('\n').map(l => l.trim()).filter(l => l);
        
        // Processar linhas para extrair informações
        let eventoEncontrado = false;
        let selecaoEncontrada = false;
        
        for (let i = 0; i < linhas.length; i++) {
            const linha = linhas[i];
            
            // 1. Capturar tipo de aposta (contém "Futebol :" ou similar)
            if (linha.includes('Futebol') && linha.includes(':') && linha.includes('Partida')) {
                dados.push('TIPO:' + linha);
                continue;
            }
            
            // 2. Capturar evento (contém "vs." ou "vs")
            if (!eventoEncontrado && (linha.includes(' vs. ') || linha.includes(' vs '))) {
                dados.push('EVENTO:' + linha);
                eventoEncontrado = true;
                continue;
            }
            
            // 3. Capturar seleção (Asian, handicap, Over/Under)
            if (!selecaoEncontrada && 
                (linha.includes('(Asian)') || linha.includes('Asian')) ||
                (linha.includes('Over ') || linha.includes('Under '))) {
                dados.push('SELECAO:' + linha);
                selecaoEncontrada = true;
                continue;
            }
            
            // 4. Capturar linha "Todos Os Agentes De Apostas"
            if (linha.includes('Todos Os Agentes De Apostas') || linha === 'Todos Os Agentes De Apostas') {
                // As próximas linhas contêm TOTAL, MÉDIA, MELHOR
                // Coletar próximas 6-8 valores numéricos
                let valores = [];
                for (let j = i + 1; j < Math.min(i + 10, linhas.length); j++) {
                    const val = linhas[j].replace(/[,$]/g, '').trim();
                    if (/^[\d.]+$/.test(val)) {
                        valores.push(val);
                    }
                    if (valores.length >= 6) break;
                }
                if (valores.length >= 6) {
                    // Formato: ODD_TOTAL|VALOR_TOTAL|ODD_MEDIA|VALOR_MEDIA|ODD_MELHOR|VALOR_MELHOR
                    dados.push('TOTAL:' + valores[0] + '|' + valores[1] + '|' + valores[2] + '|' + valores[3] + '|' + valores[4] + '|' + valores[5]);
                }
                continue;
            }
            
            // 5. Capturar casas de apostas individuais
            const casas = ['3et', '4casters', 'bdaq', 'betdex', 'bf', 'molly', 'pin', 'sbo', 'isn'];
            for (let casa of casas) {
                // Verificar se linha começa com nome da casa (case insensitive)
                if (linha.toLowerCase() === casa.toLowerCase() || 
                    linha.toLowerCase().startsWith(casa.toLowerCase())) {
                    
                    // Coletar valores das próximas linhas
                    let valores = [];
                    for (let j = i + 1; j < Math.min(i + 10, linhas.length); j++) {
                        const val = linhas[j].replace(/[,$]/g, '').trim();
                        if (/^[\d.]+$/.test(val)) {
                            valores.push(val);
                        }
                        // Parar se encontrar outra casa ou texto não numérico longo
                        if (casas.some(c => linhas[j].toLowerCase().startsWith(c)) && linhas[j] !== linha) {
                            break;
                        }
                        if (valores.length >= 6) break;
                    }
                    
                    if (valores.length >= 2) {
                        dados.push(casa.toUpperCase() + ':' + valores.join('|'));
                    }
                    break;
                }
            }
        }
        
        // Se não extraiu dados estruturados, retornar texto limpo
        if (dados.length < 2) {
            dados = ['RAW'];
            dados = dados.concat(linhas.filter(l => l.length < 100));
        }
        
        return dados.join('\n');
        
    } catch (e) {
        return "ERRO:" + e.message;
    }
}

// Executar
extrairOdds();
