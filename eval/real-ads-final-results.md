# Final result: real ads and golden set, worst of 3 (app-0.3.2, 2026-11-16)

Real ads: golden-set/real-ads-v9.jsonl (62 ads, 58 counted: 12 Skip, 5 Be careful, 41 No red flags; 4 Unsure excluded). Runs: eval/real-ads-run5-7.jsonl.
Golden set: v0.15 (65 ads). Runs: eval/run7-9-pay-fit-fix.jsonl.

| Measure | Run 1 | Run 2 | Run 3 | Worst of 3 | Target | Pass? |
|---|---|---|---|---|---|---|
| Real ads: missed scams | 0 of 17 | 0 of 17 | 0 of 17 | 0 of 17 | 0% | Yes |
| Real ads: false alarms (normal -> Skip) | 0 of 41 | 0 of 41 | 0 of 41 | 0 of 41 | under 5% | Yes |
| Real ads: normal ads given Be careful | 0 | 0 | 0 | 0 | - | - |
| Real ads: Skip accuracy | 8 of 12 | 8 of 12 | 8 of 12 | 8 of 12 | none set | - |
| Golden set: accepted | 65 of 65 | 65 of 65 | 65 of 65 | 65 of 65 | - | Yes |
| Golden set: missed scams / false alarms | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | - | Yes |

Every real ad got the same verdict in all 3 runs. Only GS-R15 varied (Skip, Be careful, Be careful; both accepted).

Ads I labelled Skip that get Be careful (still warned): RA10 (eBay, daily pay, Telegram), RA11 (Company XXX), RA12 (multi-level marketing; Be careful judged fair), RA61 (goo.gl message).

Caveats: the fixes were made after seeing these ads, so this is not an unseen test. A fresh batch of real ads would confirm it.
Cost of the 6 runs: about $1.60.
