# Parar H3BUP_vNext (live)

## Método preferido (reversível)
Kill-switch do bridge: `logs/bridge_risk_params.json` → `disable_back=true`.

O bridge recarrega ~a cada 5s e deixa de submeter Back.

```bash
sudo -u betbot /home/betbot/Bets/betinasia_bot/ops/h3bup_stop_live.sh
```

## Confirmação
- `disable_back: true` no JSON
- Novas tentativas com `disabled_back` / `operational_disabled_back`
- Sem novos `LIVE_OK` H3BUP em `logs/executor_live.jsonl` após o stop

## Não tocar
- accounting, daily report, CLV workers (analytics)
- `EXECUTOR_ALLOW_LIVE` global (afeta tudo)
- policy patches / H3BUP_VNEXT_POLICY_LOCK

## Retomar (só se decidido)
```bash
sudo -u betbot /home/betbot/Bets/betinasia_bot/ops/h3bup_resume_live.sh
```
