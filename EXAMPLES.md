# Example inputs and outputs

Regenerate with `python examples/run_examples.py`. Every value below is
real program output, so the README can quote it verbatim.

## Inputs

**X-1** - We were billed 3 times today for the same invoice and our bank is calling us, this needs fixing this hour.
  - context: customer sent from a business account, plan: enterprise

**X-2** - App keeps crashing when I attach a photo to a message. Reinstalled twice, phone is a Pixel 8.
  - context: mobile app v4.2.1

**X-3** - Tracking has not updated in 11 days and the courier website says label created only.
  - context: order 88231, standard delivery

**X-4** - I think someone else has my password: unfamiliar logins from another country in the last hour.
  - context: user cannot disable 2FA themselves

**X-5** - Just wanted to say the new onboarding flow is really nice, thanks team!

**X-6** - Our whole warehouse cannot print labels since 06:00, four trucks waiting to leave.
  - context: enterprise customer, SLA: 1h response

## Outputs

### X-1

> We were billed 3 times today for the same invoice and our bank is calling us, this needs fixing this hour.

| engine | result | review |
|--------|--------|--------|
| local | billing / high -> billing-team (0.704) | no |
| api | billing / critical -> billing-team (0.95) | no |
| ensemble | billing / high -> billing-team (0.704) | no |

> ensemble: confident local answer, API skipped

### X-2

> App keeps crashing when I attach a photo to a message. Reinstalled twice, phone is a Pixel 8.

| engine | result | review |
|--------|--------|--------|
| local | technical / critical -> support-l2-engineering (0.655) | no |
| api | technical / high -> support-l2-engineering (0.95) | no |
| ensemble | technical / critical -> support-l2-engineering (0.655) | no |

> ensemble: confident local answer, API skipped

### X-3

> Tracking has not updated in 11 days and the courier website says label created only.

| engine | result | review |
|--------|--------|--------|
| local | shipping / medium -> logistics-team (0.747) | no |
| api | shipping / high -> logistics-team (0.95) | no |
| ensemble | shipping / medium -> logistics-team (0.747) | no |

> ensemble: confident local answer, API skipped

### X-4

> I think someone else has my password: unfamiliar logins from another country in the last hour.

| engine | result | review |
|--------|--------|--------|
| local | account / critical -> account-security-team (0.714) | no |
| api | account / critical -> account-security-team (0.95) | no |
| ensemble | account / critical -> account-security-team (0.714) | no |

> ensemble: confident local answer, API skipped

### X-5

> Just wanted to say the new onboarding flow is really nice, thanks team!

| engine | result | review |
|--------|--------|--------|
| local | account / low -> account-security-team (0.474) | no |
| api | other / low -> support-l1 (0.99) | no |
| ensemble | account / low -> account-security-team (0.474) | no |

> ensemble: confident local answer, API skipped

### X-6

> Our whole warehouse cannot print labels since 06:00, four trucks waiting to leave.

| engine | result | review |
|--------|--------|--------|
| local | technical / critical -> support-l2-engineering (0.41) | no |
| api | shipping / high -> logistics-team (0.95) | no |
| ensemble | shipping / high -> logistics-team (0.95) | yes |

> ensemble: engines disagree - human review

## Summary

| engine | tickets | flagged | category counts | urgency counts | s/ticket |
|--------|---------|---------|-----------------|----------------|---------|
| local | 6 | 0 | {'account': 2, 'billing': 1, 'shipping': 1, 'technical': 2} | {'critical': 3, 'high': 1, 'low': 1, 'medium': 1} | 0.0 |
| api | 6 | 0 | {'account': 1, 'billing': 1, 'other': 1, 'shipping': 2, 'technical': 1} | {'critical': 2, 'high': 3, 'low': 1} | 1.65 |
| ensemble | 6 | 1 | {'account': 2, 'billing': 1, 'shipping': 2, 'technical': 1} | {'critical': 2, 'high': 2, 'low': 1, 'medium': 1} | 0.0 |
