# Unseen test: real ads batch 2 (27 ads, app-0.3.2, 2026-11-16)

Data: golden-set/real-ads-unseen-v2.jsonl (RB01-RB27; 25 counted: 10 Skip, 15 No red flags; 2 Unsure excluded). The app was NOT changed for this batch before the run. Runs: eval/unseen-v2-run1-3.jsonl. Cost about $0.41.

| Measure | Run 1 | Run 2 | Run 3 | Worst of 3 | Target | Pass? |
|---|---|---|---|---|---|---|
| Missed scams (my Skip -> app No red flags) | 2 of 10 | 2 of 10 | 2 of 10 | 2 of 10 (20%) | 0% | No |
| False alarms (my No red flags -> app Skip) | 0 of 15 | 0 of 15 | 0 of 15 | 0 of 15 | under 5% | Yes |
| False alarms, pooled with batch 1 (worst of 3) | | | | 0 of 56 | under 5% | Yes |
| Skip accuracy | 4 of 10 | 4 of 10 | 4 of 10 | 4 of 10 | none | - |
| Normal ads given Be careful | 3 | 2 | 3 | 3 | - | - |

## Missed scams (same in all 3 runs)
- RB04 (cruise ship jobs, "send your resume to staff4cruiseposittionstgmail.com (coppy & paste this e-mail address)"): only "pushes you to act fast" fired. The email is written without "@", so the free-email check can't see it, and the AI judged the paste partial, so "no employer named" doesn't fire either.
- RB09 (NFT game moderator, $850 a week, asks name and country): only "no interview" fired. Nothing in the text asks for money, ID or crypto payment; the crypto/NFT setting isn't a sign the app checks.

## Skip ads given Be careful (still warned)
- RB01, RB02, RB03 (big daily pay for minutes of online work): the AI tags them as task scams, but the quotes name no tasks, so they count only as the weak sign WEAK_PAY_NOT_FIT.
- RB06 (reshipping contract): reshipping is a strong sign (Be careful), not one of the Skip patterns.

## Normal ads given Be careful (not false alarms)
- RB20 (Accenture "expressions of interest", 2 of 3 runs), RB24 (Best & Less "leadership talent pool"), RB25 (The Reject Shop "talent pool"): the ghost-job talent-pool rule (11 Nov). I marked these No red flags.
