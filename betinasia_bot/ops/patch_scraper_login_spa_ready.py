#!/usr/bin/env python3
"""Minimal idempotent patch: wait for login SPA Loading to clear; never save session without root-session."""
from __future__ import annotations

import sys
from pathlib import Path

BEGIN = "# BEGIN ACCOUNTING_PHASE2A_LOGIN_WAIT"
END = "# END ACCOUNTING_PHASE2A_LOGIN_WAIT"

WAIT_BLOCK = f'''
            {BEGIN}
            # SPA Mollybet: botão Log In só funciona após "Loading..." desaparecer.
            try:
                import time as _t
                _t0 = _t.time()
                _max = float(__import__("os").getenv("BETINASIA_LOGIN_SPA_READY_SEC", "120"))
                while (_t.time() - _t0) < _max:
                    try:
                        _body = (await self._page.locator("body").inner_text())[:400].lower()
                    except Exception:
                        _body = ""
                    if "loading..." not in _body:
                        break
                    await self._page.wait_for_timeout(1000)
                else:
                    logger.warning("[login] SPA ainda mostra Loading... após timeout; tentando mesmo assim")
            except Exception as _e_spa:
                logger.warning(f"[login] spa ready wait error: {{str(_e_spa)[:120]}}")
            {END}
'''

SAVE_GUARD_BEGIN = "# BEGIN ACCOUNTING_PHASE2A_SAVE_GUARD"
SAVE_GUARD_END = "# END ACCOUNTING_PHASE2A_SAVE_GUARD"


def patch(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    changed = False
    if BEGIN not in text:
        anchor = "await password_input.fill(password)\n            await self._page.wait_for_timeout(500)"
        if anchor not in text:
            # try alternate
            anchor = "await password_input.fill(password)"
        if anchor not in text:
            raise SystemExit("login fill anchor not found")
        text = text.replace(anchor, anchor + "\n" + WAIT_BLOCK, 1)
        changed = True
        print("inserted login wait")

    if SAVE_GUARD_BEGIN not in text:
        # wrap save_session body
        old = '''    async def save_session(self):
        """Salva a sessão atual (cookies) para reutilização."""
        try:
            await self._context.storage_state(path=self.session_file)
            logger.info(f"Sessão salva em: {self.session_file}")
        except Exception as e:
            logger.warning(f"Erro ao salvar sessão: {e}")'''
        new = '''    async def save_session(self):
        """Salva a sessão atual (cookies) para reutilização."""
        try:
            ''' + SAVE_GUARD_BEGIN + '''
            # Nunca persistir storage_state sem root-session (evita corromper sessão partilhada).
            try:
                if not await self._has_root_session_cookie():
                    logger.warning("save_session abortado: sem root-session cookie")
                    return
            except Exception as _e_sg:
                logger.warning(f"save_session guard error: {str(_e_sg)[:120]}")
                return
            ''' + SAVE_GUARD_END + '''
            await self._context.storage_state(path=self.session_file)
            logger.info(f"Sessão salva em: {self.session_file}")
        except Exception as e:
            logger.warning(f"Erro ao salvar sessão: {e}")'''
        if old not in text:
            raise SystemExit("save_session block not found exactly")
        text = text.replace(old, new, 1)
        changed = True
        print("patched save_session guard")

    if changed:
        path.write_text(text, encoding="utf-8")
        print("patched", path)
    else:
        print("already_patched")


if __name__ == "__main__":
    patch(Path(sys.argv[1] if len(sys.argv) > 1 else "scraper/betinasia.py"))
