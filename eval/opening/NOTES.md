# opening_design: honest numbers

Split by PROJECT: dev = Sports_Complex (2 sheets, 44 q), held-out = CTR (1 sheet, 20 q).
Scoring: count/area official (exact / 2%). Text uses a lenient check: every number/code token of the gold must appear in our answer; if the gold has none, its first word must. Text is reported separately from the official score.

| split | family | n | acc | abstain |
|---|---|---|---|---|
| held-out FIRST RUN | count | 17 | 64.7% | 0% |
| held-out FIRST RUN | text | 3 | 33.3% | 33.3% |
| held-out after fixes (errors were seen, so optimistic) | count | 17 | 88.2% | 5.9% |
| held-out after fixes (errors were seen, so optimistic) | text | 3 | 100% | 0% |
| dev | area | 9 | 100% | 0% |
| dev | count | 22 | 90.9% | 9.1% (A-701 plan-view fixture counts: no schedule, abstain) |
| dev | text | 13 | 84.6% | 7.7% |

The first-run misses were: free-text row filters (product names, floor qualifiers, 'description starts with …'); dimension tokens leaking into the Count column; a unanimous lookup with no row id; and a material lookup matching the wrong column.
The fixes are general (description-text filter, Count column excluded from filters, unanimous-row lookup), but they were informed by test errors.
