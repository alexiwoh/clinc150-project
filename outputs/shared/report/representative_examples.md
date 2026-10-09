## Representative Examples

*Representative run only (single seed)*

Selection prioritizes OOS false accepts, same-domain and cross-domain confusions, and per-model coverage, subject to available curated candidates. Short-query coverage is measured independently and can overlap primary categories. Sorted by category then confidence.

Taxonomy labels are heuristics: `near_semantic_confusion` means a same-domain misclassification; `short_query_ambiguity` marks errors with at most 5 whitespace tokens. These rules do not establish semantic similarity or query ambiguity.

### Coverage in Selected Examples

- Selected examples include 12 OOS false-accept cases.
- Selected examples include 5 same-domain cases tagged `near_semantic_confusion`.
- Selected examples include 7 cross-domain confusion cases.
- Selected examples include 4 short-query tagged errors (at most 5 whitespace tokens; categories may overlap).

Related figures: `outputs/mlp/analysis/oos_error_breakdown.png`, `outputs/text_cnn/analysis/oos_error_breakdown.png`, `outputs/bilstm/analysis/oos_error_breakdown.png`, `outputs/shared/analysis/oos_error_comparison.png`

Related figures: `outputs/mlp/figures/representative_confusion_matrix.png`, `outputs/mlp/figures/representative_top_confused_pairs.png`, `outputs/text_cnn/figures/representative_confusion_matrix.png`, `outputs/text_cnn/figures/representative_top_confused_pairs.png`, `outputs/bilstm/figures/representative_confusion_matrix.png`, `outputs/bilstm/figures/representative_top_confused_pairs.png`

Related figures: `outputs/mlp/figures/representative_error_summary.png`, `outputs/text_cnn/figures/representative_error_summary.png`, `outputs/bilstm/figures/representative_error_summary.png`, `outputs/shared/analysis/cross_model_error_overlap.png`

Related figure: `outputs/shared/analysis/length_slice_comparison.png`

| Text                                                            | True Label           | Predicted             | Model        | Confidence | Category                | Annotation             |
| --------------------------------------------------------------- | -------------------- | --------------------- | ------------ | ---------- | ----------------------- | ---------------------- |
| give me the weather forecast for today                          | oos                  | weather               | BiLSTM       | 1.0000     | oos_as_inscope          | confident false accept |
| look up the conversion rate for the euro to dollar exchange     | oos                  | exchange_rate         | BiLSTM       | 1.0000     | oos_as_inscope          | confident false accept |
| what other countries speak the english language                 | oos                  | change_language       | BiLSTM       | 1.0000     | oos_as_inscope          | confident false accept |
| can you tell me what the best places are to look for a job o... | oos                  | restaurant_suggestion | BiLSTM       | 1.0000     | oos_as_inscope          | confident false accept |
| give me the weather forecast for today                          | oos                  | weather               | Text CNN     | 1.0000     | oos_as_inscope          | confident false accept |
| i need you to order a new pair of eyeglasses for me             | oos                  | order                 | BiLSTM       | 1.0000     | oos_as_inscope          | confident false accept |
| please read the text message i just received                    | oos                  | text                  | Text CNN     | 0.9997     | oos_as_inscope          | confident false accept |
| what other countries speak the english language                 | oos                  | change_language       | Text CNN     | 0.9991     | oos_as_inscope          | confident false accept |
| what's the current prevailing interest rate for mortgages in... | oos                  | interest_rate         | Text CNN     | 0.9990     | oos_as_inscope          | confident false accept |
| forward the text i just got from henry to giselle               | oos                  | text                  | Text CNN     | 0.9988     | oos_as_inscope          | confident false accept |
| ignore call                                                     | oos                  | make_call             | TF-IDF + MLP | 0.9980     | oos_as_inscope          | confident false accept |
| deny incoming phone call                                        | oos                  | make_call             | TF-IDF + MLP | 0.9972     | oos_as_inscope          | confident false accept |
| give me a recipe for tacos                                      | ingredients_list     | recipe                | BiLSTM       | 1.0000     | near_semantic_confusion | same-domain confusion  |
| what's a good recipe foe tacos                                  | ingredients_list     | recipe                | BiLSTM       | 1.0000     | near_semantic_confusion | same-domain confusion  |
| what is the next date for which i can get an oil change appo... | schedule_maintenance | oil_change_when       | BiLSTM       | 1.0000     | near_semantic_confusion | same-domain confusion  |
| what are the steps to get my rewards for my visa card           | redeem_rewards       | rewards_balance       | BiLSTM       | 0.9999     | near_semantic_confusion | same-domain confusion  |
| what have i spent things on                                     | transactions         | spending_history      | BiLSTM       | 0.9999     | near_semantic_confusion | same-domain confusion  |
| repeat what the weather will be like                            | transfer             | weather               | BiLSTM       | 1.0000     | cross_domain_confusion  | cross-domain mix-up    |
| i'd like for this person to know my location                    | share_location       | current_location      | BiLSTM       | 1.0000     | cross_domain_confusion  | cross-domain mix-up    |
| what time is it in phoenix                                      | timezone             | time                  | BiLSTM       | 1.0000     | cross_domain_confusion  | cross-domain mix-up    |
| when will my payment be deposited                               | payday               | bill_due              | BiLSTM       | 1.0000     | cross_domain_confusion  | cross-domain mix-up    |
| what is my current location                                     | share_location       | current_location      | BiLSTM       | 0.9999     | cross_domain_confusion  | cross-domain mix-up    |
| what is my current location                                     | share_location       | current_location      | Text CNN     | 0.9988     | cross_domain_confusion  | cross-domain mix-up    |
| what time is it in phoenix                                      | timezone             | time                  | Text CNN     | 0.9987     | cross_domain_confusion  | cross-domain mix-up    |
| how many minutes are involved in the preparation of curry       | cook_time            | oos                   | BiLSTM       | 1.0000     | inscope_as_oos          | false rejection        |

Total curated examples: 60; selected for report: 25 (outputs/shared/analysis/curated_report_examples.json).
