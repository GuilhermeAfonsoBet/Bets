"""Script para fazer logout do BetinAsia e limpar a sessão."""
import asyncio
import os
from pathlib import Path

async def logout():
    session_file = Path("betinasia_session.json")
    
    if session_file.exists():
        # Remove o arquivo de sessão
        session_file.unlink()
        print("✓ Sessão removida com sucesso!")
        print("  O bot precisará fazer login novamente na próxima execução.")
    else:
        print("ℹ Nenhuma sessão ativa encontrada.")
    
    print("\nVocê pode navegar manualmente no site agora.")

if __name__ == "__main__":
    asyncio.run(logout())
