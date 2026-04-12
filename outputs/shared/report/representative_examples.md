## Representative Examples

*Representative run only (single seed)*

Examples selected to cover key error patterns: at least 3 OOS false accepts, 3 semantic confusions, 3 cross-domain confusions, 3 short-query ambiguity, and 2+ examples per model. Sorted by category then confidence.

### Coverage in Selected Examples

- Selected examples include 12 OOS false-accept cases.
- Selected examples include 7 within-domain semantic-confusion cases.
- Selected examples include 5 cross-domain confusion cases.
- Selected examples include 0 short-query ambiguity cases.

Related figures: `outputs/mlp/analysis/oos_error_breakdown.png`, `outputs/text_cnn/analysis/oos_error_breakdown.png`, `outputs/bilstm/analysis/oos_error_breakdown.png`, `outputs/shared/analysis/oos_error_comparison.png`

Related figures: `outputs/mlp/figures/representative_confusion_matrix.png`, `outputs/mlp/figures/representative_top_confused_pairs.png`, `outputs/text_cnn/figures/representative_confusion_matrix.png`, `outputs/text_cnn/figures/representative_top_confused_pairs.png`, `outputs/bilstm/figures/representative_confusion_matrix.png`, `outputs/bilstm/figures/representative_top_confused_pairs.png`

Related figures: `outputs/mlp/figures/representative_error_summary.png`, `outputs/text_cnn/figures/representative_error_summary.png`, `outputs/bilstm/figures/representative_error_summary.png`, `outputs/shared/analysis/cross_model_error_overlap.png`

Related figure: `outputs/shared/analysis/length_slice_comparison.png`

| Text                                                            | True Label                | Predicted            | Model        | Confidence | Category                | Annotation             |
| --------------------------------------------------------------- | ------------------------- | -------------------- | ------------ | ---------- | ----------------------- | ---------------------- |
| what other countries speak the english language                 | oos                       | change_language      | BiLSTM       | 1.0000     | oos_as_inscope          | confident false accept |
| how many calories does doing 20 situps burn                     | oos                       | calories             | Text CNN     | 1.0000     | oos_as_inscope          | confident false accept |
| give me the weather forecast for today                          | oos                       | weather              | Text CNN     | 1.0000     | oos_as_inscope          | confident false accept |
| look up the conversion rate for the euro to dollar exchange     | oos                       | exchange_rate        | BiLSTM       | 1.0000     | oos_as_inscope          | confident false accept |
| give me the weather forecast for today                          | oos                       | weather              | BiLSTM       | 1.0000     | oos_as_inscope          | confident false accept |
| how many calories does jumping up and down burn                 | oos                       | calories             | Text CNN     | 0.9999     | oos_as_inscope          | confident false accept |
| check the status of my amazon orders for me                     | oos                       | order_status         | BiLSTM       | 0.9999     | oos_as_inscope          | confident false accept |
| call an uber to take me to the closest grocery store            | oos                       | uber                 | Text CNN     | 0.9999     | oos_as_inscope          | confident false accept |
| when should i remove my snow tires                              | oos                       | tire_change          | BiLSTM       | 0.9999     | oos_as_inscope          | confident false accept |
| what's the current prevailing interest rate for mortgages in... | oos                       | interest_rate        | Text CNN     | 0.9995     | oos_as_inscope          | confident false accept |
| ignore call                                                     | oos                       | make_call            | TF-IDF + MLP | 0.9980     | oos_as_inscope          | confident false accept |
| deny incoming phone call                                        | oos                       | make_call            | TF-IDF + MLP | 0.9972     | oos_as_inscope          | confident false accept |
| have i told you to add washing dishes to my todo list           | todo_list                 | todo_list_update     | BiLSTM       | 0.9999     | near_semantic_confusion | semantic overlap       |
| thanks for your help, goodbye!                                  | goodbye                   | thank_you            | BiLSTM       | 0.9999     | near_semantic_confusion | semantic overlap       |
| give me a recipe for tacos                                      | ingredients_list          | recipe               | BiLSTM       | 0.9999     | near_semantic_confusion | semantic overlap       |
| have i added my doctor's appointment to my calendar             | calendar                  | calendar_update      | BiLSTM       | 0.9999     | near_semantic_confusion | semantic overlap       |
| how can i request a new credit card                             | replacement_card_duration | new_card             | BiLSTM       | 0.9999     | near_semantic_confusion | semantic overlap       |
| who is responsible for your employment                          | who_do_you_work_for       | who_made_you         | Text CNN     | 0.9986     | near_semantic_confusion | semantic overlap       |
| i'm trying to raise my credit score can you tell me what it ... | credit_score              | improve_credit_score | Text CNN     | 0.9985     | near_semantic_confusion | semantic overlap       |
| what time is it in phoenix                                      | timezone                  | time                 | BiLSTM       | 1.0000     | cross_domain_confusion  | cross-domain mix-up    |
| what is my current location                                     | share_location            | current_location     | BiLSTM       | 1.0000     | cross_domain_confusion  | cross-domain mix-up    |
| repeat what the weather will be like                            | transfer                  | weather              | BiLSTM       | 1.0000     | cross_domain_confusion  | cross-domain mix-up    |
| i want to be reminded to pay the electric bill                  | reminder_update           | pay_bill             | BiLSTM       | 0.9999     | cross_domain_confusion  | cross-domain mix-up    |
| i don't want to forget to call mom                              | reminder_update           | make_call            | BiLSTM       | 0.9998     | cross_domain_confusion  | cross-domain mix-up    |
| what is the price of bluetooth speakers on amazon               | order                     | oos                  | BiLSTM       | 0.9995     | inscope_as_oos          | false rejection        |

Total curated examples: 60; selected for report: 25 (outputs/shared/analysis/curated_report_examples.json).
