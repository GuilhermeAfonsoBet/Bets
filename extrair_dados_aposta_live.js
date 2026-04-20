// Script para extrair dados do painel de apostas live do BetInAsia
// Para uso com Power Automate Desktop (PAD)

function extrairDadosApostaLive() {
    try {
        // Encontrar o painel Group (painel lateral de apostas)
        // Usando as classes identificadas no seletor: _62e8462d _6db2ad38 _3c03fdb4 _34888e3f
        let painelGroup = document.querySelector('[class*="_62e8462d"][class*="_6db2ad38"]');
        
        // Alternativa: buscar pelo conteúdo característico do painel
        if (!painelGroup) {
            painelGroup = document.querySelector('[id="root"] [class*="_5f82213e"]');
        }
        
        // Outra alternativa: buscar pelo título "Group"
        if (!painelGroup) {
            const groupHeaders = document.querySelectorAll('*');
            for (let el of groupHeaders) {
                if (el.textContent && el.textContent.includes('Todos Os Agentes De Apostas')) {
                    painelGroup = el.closest('[class*="_62e8462d"]') || el.parentElement.parentElement.parentElement;
                    break;
                }
            }
        }

        if (!painelGroup) {
            return "ERRO: Painel de apostas não encontrado";
        }

        // Array para armazenar as linhas de dados
        let linhasDados = [];

        // 1. Extrair tipo de aposta e evento (ex: "Futebol : Final Da Partida")
        const tipoApostaEl = painelGroup.querySelector('[class*="betType"], [class*="market"]') ||
                           Array.from(painelGroup.querySelectorAll('span, div')).find(el => 
                               el.textContent && el.textContent.includes('Futebol') && el.textContent.includes(':'));
        
        if (tipoApostaEl) {
            linhasDados.push(tipoApostaEl.textContent.trim());
        }

        // 2. Extrair evento (ex: "AEP Paphos FC vs. Slavia Praha")
        const eventoEl = Array.from(painelGroup.querySelectorAll('span, div')).find(el => 
            el.textContent && el.textContent.includes(' vs. ') || el.textContent.includes(' vs '));
        
        if (eventoEl) {
            linhasDados.push(eventoEl.textContent.trim());
        }

        // 3. Extrair seleção da aposta (ex: "Slavia Praha +0.5 (Asian)")
        const selecaoEl = Array.from(painelGroup.querySelectorAll('h3, h4, [class*="selection"], [class*="bet-name"]')).find(el =>
            el.textContent && (el.textContent.includes('+') || el.textContent.includes('-') || el.textContent.includes('Asian')));
        
        if (selecaoEl) {
            linhasDados.push(selecaoEl.textContent.trim());
        }

        // 4. Extrair dados das casas de apostas
        // Procurar pela tabela ou lista de odds
        const linhasOdds = painelGroup.querySelectorAll('[class*="row"], tr, [class*="agent"], [class*="book"]');
        
        // Lista de nomes de casas conhecidas
        const casasConhecidas = ['3et', '4casters', 'bdaq', 'betdex', 'bf', 'molly', 'pin', 'betfair', 'pinnacle'];
        
        for (let linha of linhasOdds) {
            const textoLinha = linha.textContent.trim();
            
            // Verificar se a linha contém uma casa de apostas conhecida
            for (let casa of casasConhecidas) {
                if (textoLinha.toLowerCase().includes(casa.toLowerCase())) {
                    // Extrair os números (odds e valores)
                    const numeros = textoLinha.match(/[\d.,]+/g);
                    if (numeros && numeros.length >= 2) {
                        // Formato: Casa | Odd1 | Valor1 | Odd2 | Valor2 | Odd3 | Valor3
                        linhasDados.push(`${casa}|${numeros.join('|')}`);
                    }
                    break;
                }
            }
        }

        // 5. Se não encontrou dados estruturados, tentar extrair todo o texto do painel
        if (linhasDados.length < 3) {
            // Fallback: extrair texto completo organizado
            const todosElementos = painelGroup.querySelectorAll('span, div, td');
            let textoCompleto = [];
            
            for (let el of todosElementos) {
                if (el.children.length === 0) { // Apenas elementos folha
                    const texto = el.textContent.trim();
                    if (texto && texto.length > 0 && texto.length < 100) {
                        textoCompleto.push(texto);
                    }
                }
            }
            
            // Remover duplicatas mantendo ordem
            textoCompleto = [...new Set(textoCompleto)];
            linhasDados = textoCompleto;
        }

        // Retornar dados separados por quebra de linha
        return linhasDados.join('\n');

    } catch (erro) {
        return "ERRO: " + erro.message;
    }
}

// Executar e retornar resultado
extrairDadosApostaLive();
