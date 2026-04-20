// Script JS compatível com o fluxo PAD existente
// Retorna o texto bruto do painel Group no mesmo formato que o seletor UIAutomation
// Para uso com "Executar função JavaScript na página da Web" do PAD

function extrairDadosApostaLive() {
    try {
        // ===== LOCALIZAR O PAINEL GROUP =====
        // Mesmas classes do seletor PAD: _62e8462d _6db2ad38 _3c03fdb4 _34888e3f
        // ou _4e93ff49 _1440e97e _15f095be _4befa0c9
        
        let painelGroup = null;
        
        // Método 1: Buscar pelas classes do seletor PAD
        painelGroup = document.querySelector('[class*="_62e8462d"][class*="_6db2ad38"]');
        
        if (!painelGroup) {
            painelGroup = document.querySelector('[class*="_4e93ff49"][class*="_1440e97e"]');
        }
        
        // Método 2: Buscar pelo container que tem a tabela de odds
        if (!painelGroup) {
            const divs = document.querySelectorAll('div');
            for (let div of divs) {
                const texto = div.innerText || '';
                // Procurar pelo painel que contém "Todos Os Agentes De Apostas" E valores de odds
                if (texto.includes('Todos Os Agentes De Apostas') && 
                    texto.includes('MELHOR') &&
                    /\d+[.,]\d{2,3}/.test(texto)) {
                    
                    // Verificar se não é um container muito grande (queremos apenas o painel lateral)
                    if (texto.length < 5000) {
                        painelGroup = div;
                        break;
                    }
                }
            }
        }
        
        // Método 3: Buscar pelo painel lateral direito fixo
        if (!painelGroup) {
            const paineis = document.querySelectorAll('[class*="_5f82213e"]');
            for (let p of paineis) {
                if (p.innerText && p.innerText.includes('Todos Os Agentes')) {
                    painelGroup = p;
                    break;
                }
            }
        }

        if (!painelGroup) {
            return "ERRO: Painel de apostas não encontrado";
        }

        // ===== EXTRAIR TEXTO BRUTO =====
        // Retorna o innerText do elemento, que é exatamente o que o seletor UIAutomation retornaria
        // O PAD vai dividir por quebra de linha e processar com as regexes existentes
        
        let textoExtraido = painelGroup.innerText;
        
        // Limpar linhas vazias duplas e espaços excessivos
        textoExtraido = textoExtraido
            .split('\n')
            .map(linha => linha.trim())
            .filter(linha => linha.length > 0)
            .join('\n');
        
        return textoExtraido;

    } catch (erro) {
        return "ERRO: " + erro.message;
    }
}

// Executar e retornar resultado
extrairDadosApostaLive();
