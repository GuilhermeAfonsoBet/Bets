// Script de DEBUG para verificar o formato exato dos dados extraídos
// Execute este script no Console do navegador (F12) para ver a estrutura
// Isso ajuda a entender o formato atual do seletor UIAutomation

function debugExtrairDados() {
    console.log("=== INICIANDO DEBUG ===");
    
    // Tentar diferentes seletores
    const seletores = [
        '[class*="_62e8462d"][class*="_6db2ad38"]',
        '[class*="_4e93ff49"][class*="_1440e97e"]',
        '[class*="_5f82213e"][class*="_daabd7"]',
        '[class*="_62e8462d"]',
        '[class*="_4e93ff49"]'
    ];
    
    for (let sel of seletores) {
        const el = document.querySelector(sel);
        if (el) {
            console.log(`\n✓ Encontrado com seletor: ${sel}`);
            console.log(`Classes: ${el.className}`);
            console.log(`Tamanho texto: ${el.innerText.length} caracteres`);
        }
    }
    
    // Buscar pelo conteúdo
    console.log("\n=== BUSCANDO POR CONTEÚDO ===");
    
    let painelEncontrado = null;
    const divs = document.querySelectorAll('div');
    
    for (let div of divs) {
        const texto = div.innerText || '';
        if (texto.includes('Todos Os Agentes De Apostas') && 
            texto.includes('MELHOR') &&
            texto.length < 5000 &&
            texto.length > 100) {
            
            console.log(`\n✓ Painel candidato encontrado`);
            console.log(`Classes: ${div.className}`);
            console.log(`Tamanho: ${texto.length} caracteres`);
            
            if (!painelEncontrado || texto.length < painelEncontrado.innerText.length) {
                painelEncontrado = div;
            }
        }
    }
    
    if (!painelEncontrado) {
        console.log("✗ Nenhum painel encontrado!");
        return "ERRO: Painel não encontrado";
    }
    
    // Mostrar texto extraído
    console.log("\n=== TEXTO EXTRAÍDO (BRUTO) ===");
    const textobruto = painelEncontrado.innerText;
    console.log(textobruto);
    
    console.log("\n=== LINHAS DIVIDIDAS ===");
    const linhas = textobruto.split('\n').map(l => l.trim()).filter(l => l);
    linhas.forEach((linha, idx) => {
        console.log(`[${idx}] "${linha}"`);
    });
    
    console.log("\n=== ANÁLISE DAS LINHAS ===");
    const regexOdd = /^\s*\d+(?:[.,]\d{2,3})?\s*$/;
    const regexStake = /^\s*(?:R\$|\$)\s*\d+(?:[.,]\d{2,3})?\s*$/;
    
    linhas.forEach((linha, idx) => {
        let tipo = "TEXTO";
        if (regexOdd.test(linha)) tipo = "ODD";
        else if (regexStake.test(linha)) tipo = "STAKE";
        else if (linha.toLowerCase().includes('todos') && linha.toLowerCase().includes('apostas')) tipo = "AGENTES";
        else if (['3et', '4casters', 'bdaq', 'betdex', 'bf', 'molly', 'pin', 'sbo'].some(c => linha.toLowerCase() === c)) tipo = "CASA";
        
        console.log(`[${idx}] ${tipo.padEnd(8)} → "${linha}"`);
    });
    
    console.log("\n=== FIM DEBUG ===");
    
    // Retornar texto formatado para PAD
    return linhas.join('\n');
}

// Executar
debugExtrairDados();
