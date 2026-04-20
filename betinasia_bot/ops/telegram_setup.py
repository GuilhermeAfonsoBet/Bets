# -*- coding: utf-8 -*-
"""
Ajuda a obter TELEGRAM_CHAT_ID para alertas do bot.

Pré-requisito:
- crie um bot no @BotFather e coloque TELEGRAM_BOT_TOKEN no .env
- no seu Telegram (telefone), abra o bot e mande /start (ou qualquer mensagem)

Depois rode:
  python3 -m ops.telegram_setup

Ele tentará ler getUpdates e imprimir os chat_id possíveis.
"""

from __future__ import annotations

import json
import os
import sys
from urllib.request import urlopen


def main() -> int:
    token = os.getenv("TELEGRAM_BOT_TOKEN") or ""
    if not token:
        print("ERRO: TELEGRAM_BOT_TOKEN não está definido no ambiente/.env.")
        return 2

    url = f"https://api.telegram.org/bot{token}/getUpdates"
    try:
        with urlopen(url, timeout=10) as resp:
            payload = json.loads(resp.read().decode("utf-8", errors="ignore"))
    except Exception as e:
        print(f"ERRO: falha ao chamar getUpdates: {e}")
        return 2

    if not isinstance(payload, dict) or "result" not in payload:
        print("ERRO: resposta inesperada do Telegram.")
        print(payload)
        return 2

    updates = payload.get("result") or []
    if not updates:
        print("Nenhum update encontrado.")
        print("Abra o bot no Telegram e mande /start, depois rode este comando novamente.")
        return 1

    chats = []
    for u in updates:
        msg = u.get("message") or u.get("edited_message") or {}
        chat = msg.get("chat") or {}
        chat_id = chat.get("id")
        chat_type = chat.get("type")
        title = chat.get("title") or ""
        username = chat.get("username") or ""
        text = (msg.get("text") or "")[:80]
        if chat_id is None:
            continue
        chats.append((chat_id, chat_type, title, username, text))

    # dedup preservando ordem
    seen = set()
    uniq = []
    for c in chats:
        if c[0] in seen:
            continue
        seen.add(c[0])
        uniq.append(c)

    print("Chat(s) encontrado(s):")
    for chat_id, chat_type, title, username, text in uniq:
        label = title or (f"@{username}" if username else "")
        print(f"- chat_id={chat_id} type={chat_type} {label} last_text='{text}'")

    print("\nDefina no .env:")
    print(f"TELEGRAM_CHAT_ID={uniq[0][0]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

