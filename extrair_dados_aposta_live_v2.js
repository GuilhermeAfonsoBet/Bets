// Script para extrair dados do painel "Group" de apostas live do BetInAsia
// Formato de saída: texto com quebras de linha para PAD processar com "Dividir o texto"
// Versão 2 - Otimizado para a estrutura específica do site

function extrairDadosApostaLive() {
    try {
        let resultado = [];
        
        // ===== LOCALIZAR O PAINEL GROUP =====
        // O painel tem classe similar a: _62e8462d _6db2ad38 _3c03fdb4 _34888e3f
        // ou está dentro de um elemento com classe _5f82213e _daabd7
        
        let painelGroup = null;
        
        // Método 1: Buscar pelo seletor de classes do PAD
        painelGroup = document.querySelector('[class*="_62e8462d"]');
        
        // Método 2: Buscar pelo texto "Todos Os Agentes De Apostas"
        if (!painelGroup) {
            const elementos = document.querySelectorAll('*');
            for (let el of elementos) {
                if (el.textContent === 'Todos Os Agentes De Apostas') {
                    // Subir na árvore até encontrar o container principal
                    painelGroup = el.closest('[class*="_4e93ff49"]') || 
                                 el.closest('[class*="_62e8462d"]') ||
                                 el.parentElement?.parentElement?.parentElement?.parentElement?.parentElement;
                    break;
                }
            }
        }
        
        // Método 3: Buscar pelo painel lateral direito (geralmente fixo à direita)
        if (!painelGroup) {
            const paineis = document.querySelectorAll('div[style*="position: fixed"], div[style*="right"]');
            for (let p of paineis) {
                if (p.textContent.includes('Agentes De Apostas') || p.textContent.includes('MELHOR')) {
                    painelGroup = p;
                    break;
                }
            }
        }

        if (!painelGroup) {
            return "ERRO: Painel Group não encontrado. Verifique se o painel de apostas está aberto.";
        }

        // ===== EXTRAIR DADOS DA APOSTA =====
        
        // 1. Tipo de aposta (ex: "Futebol : Final Da Partida")
        const todosTextos = painelGroup.innerText.split('\n').filter(t => t.trim());
        
        for (let texto of todosTextos) {
            texto = texto.trim();
            
            // Identificar tipo de aposta (contém ":")
            if (texto.includes('Futebol') && texto.includes(':')) {
                resultado.push('TIPO|' + texto);
                continue;
            }
            
            // Identificar evento (contém "vs" ou "vs.")
            if (texto.includes(' vs. ') || texto.includes(' vs ') || texto.includes(' x ')) {
                resultado.push('EVENTO|' + texto);
                continue;
            }
            
            // Identificar seleção (contém handicap Asian ou Over/Under)
            if ((texto.includes('(Asian)') || texto.includes('Asian')) && 
                (texto.includes('+') || texto.includes('-'))) {
                resultado.push('SELECAO|' + texto);
                continue;
            }
            
            // Identificar Over/Under
            if (texto.includes('Over') || texto.includes('Under')) {
                resultado.push('SELECAO|' + texto);
                continue;
            }
        }

        // ===== EXTRAIR TABELA DE ODDS =====
        
        // Procurar elementos que contêm as casas de apostas e suas odds
        const casas = ['3et', '4casters', 'bdaq', 'betdex', 'bf', 'molly', 'pin', 'betfair', 'sbo', 'isn'];
        
        // Procurar por linhas da tabela
        const linhas = painelGroup.querySelectorAll('[class*="row"], [class*="agent"], tr, [class*="_"]');
        
        for (let linha of linhas) {
            const textoLinha = linha.innerText || linha.textContent;
            const textoLimpo = textoLinha.replace(/\s+/g, ' ').trim();
            
            // Verificar se é linha de "Todos Os Agentes"
            if (textoLimpo.includes('Todos Os Agentes')) {
                // Extrair: TOTAL, MÉDIA, MELHOR
                const numeros = textoLimpo.match(/[\d.,]+/g);
                if (numeros && numeros.length >= 3) {
                    // Formato esperado: odds e valores alternados
                    resultado.push('TODOS|' + numeros.slice(0, 6).join('|'));
                }
                continue;
            }
            
            // Verificar se é linha de casa de apostas
            for (let casa of casas) {
                if (textoLimpo.toLowerCase().startsWith(casa.toLowerCase()) ||
                    textoLimpo.toLowerCase().includes(casa.toLowerCase() + ' ') ||
                    textoLimpo.toLowerCase() === casa.toLowerCase()) {
                    
                    // Extrair números da linha
                    const numeros = textoLimpo.match(/[\d.,]+/g);
                    if (numeros && numeros.length >= 2) {
                        // Formato: CASA|ODD_TOTAL|VALOR_TOTAL|ODD_MEDIA|VALOR_MEDIA|ODD_MELHOR|VALOR_MELHOR
                        resultado.push(casa.toUpperCase() + '|' + numeros.join('|'));
                    }
                    break;
                }
            }
        }

        // ===== FALLBACK: EXTRAIR TEXTO ESTRUTURADO =====
        if (resultado.length < 2) {
            // Se não conseguiu extrair de forma estruturada, retornar texto limpo
            resultado = ['TEXTO_BRUTO'];
            const textoBruto = painelGroup.innerText;
            const linhasTexto = textoBruto.split('\n')
                .map(l => l.trim())
                .filter(l => l.length > 0 && l.length < 200);
            resultado = resultado.concat(linhasTexto);
        }

        // Retornar resultado separado por quebras de linha
        return resultado.join('\n');

    } catch (erro) {
        return "ERRO|" + erro.message;
    }
}

// Executar e retornar resultado
extrairDadosApostaLive();
