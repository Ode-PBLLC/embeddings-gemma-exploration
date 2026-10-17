# Modality gap and retrieval policy

## mode=raw, image_input=image+title: 12/15 mixed walls (need 12)

| query                         |   top1 |   aic |   nasa |   wiki | mixed   |   video |
|:------------------------------|-------:|------:|-------:|-------:|:--------|--------:|
| loneliness                    |  0.760 |     6 |      6 |      0 | True    |       0 |
| the color blue                |  0.735 |     0 |      5 |      7 | True    |       0 |
| storms and turbulent skies    |  0.759 |     1 |     11 |      0 | True    |       0 |
| the night sky                 |  0.761 |     0 |     12 |      0 | False   |       0 |
| light on water                |  0.716 |     4 |      8 |      0 | True    |       0 |
| red planet dust               |  0.732 |     0 |      9 |      3 | True    |       0 |
| a portrait of a woman         |  0.793 |    12 |      0 |      0 | False   |       0 |
| spirals                       |  0.779 |     0 |      9 |      3 | True    |       0 |
| gold and luxury               |  0.709 |    11 |      1 |      0 | True    |       0 |
| explorers and voyages         |  0.707 |     8 |      3 |      1 | True    |       2 |
| fire and heat                 |  0.711 |     1 |      9 |      2 | True    |       3 |
| gardens in bloom              |  0.714 |    11 |      1 |      0 | True    |       0 |
| the ocean seen from far above |  0.733 |     0 |     12 |      0 | False   |       4 |
| ancient gods                  |  0.730 |    11 |      0 |      1 | True    |       0 |
| machines that fly             |  0.717 |     0 |      9 |      3 | True    |       5 |

**Chosen:** mode=`raw`, image_input=`image+title`.

## Weak threshold
- on-topic top-1 similarity: min 0.707, p10 0.709, median 0.732
- off-topic top-1 similarity: p90 0.626, median 0.593, max 0.666
- threshold = midpoint(p10 on, p90 off) = **0.6677**
- off-topic flagged weak: 10/10 (need >= 8); on-topic flagged weak: 0/15 (need <= 2)
- result: PASS

Off-topic top-1: 'quarterly tax filing deadlines'=0.604, 'how to reset my router password'=0.578, 'best mortgage refinance rates'=0.603, 'python list comprehension syntax'=0.583, 'fantasy football waiver wire advice'=0.583, 'symptoms of a sinus infection'=0.607, 'unsubscribe from this mailing list'=0.621, 'kubernetes pod crashloopbackoff'=0.666, 'coupon code for pizza delivery'=0.579, 'spreadsheet pivot table formula'=0.581
