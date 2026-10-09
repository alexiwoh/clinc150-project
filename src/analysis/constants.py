"""Analysis-specific constants: CLINC150 domain mapping, bin edges, output filenames."""

from __future__ import annotations

from src.constants import SHORT_QUERY_TOKEN_THRESHOLD as SHORT_QUERY_TOKEN_THRESHOLD

# ---------------------------------------------------------------------------
# CLINC150 intent domain mapping (10 domains x 15 intents)
# Source: https://github.com/clinc/oos-eval/blob/master/data/domains.json
# ---------------------------------------------------------------------------

CLINC150_INTENT_DOMAINS: dict[str, str] = {
    # banking
    "freeze_account": "banking",
    "routing": "banking",
    "pin_change": "banking",
    "bill_due": "banking",
    "pay_bill": "banking",
    "account_blocked": "banking",
    "interest_rate": "banking",
    "min_payment": "banking",
    "bill_balance": "banking",
    "transfer": "banking",
    "order_checks": "banking",
    "balance": "banking",
    "spending_history": "banking",
    "transactions": "banking",
    "report_fraud": "banking",
    # credit_cards
    "replacement_card_duration": "credit_cards",
    "expiration_date": "credit_cards",
    "damaged_card": "credit_cards",
    "improve_credit_score": "credit_cards",
    "report_lost_card": "credit_cards",
    "card_declined": "credit_cards",
    "credit_limit_change": "credit_cards",
    "apr": "credit_cards",
    "redeem_rewards": "credit_cards",
    "credit_limit": "credit_cards",
    "rewards_balance": "credit_cards",
    "application_status": "credit_cards",
    "credit_score": "credit_cards",
    "new_card": "credit_cards",
    "international_fees": "credit_cards",
    # kitchen_and_dining
    "food_last": "kitchen_and_dining",
    "confirm_reservation": "kitchen_and_dining",
    "how_busy": "kitchen_and_dining",
    "ingredients_list": "kitchen_and_dining",
    "calories": "kitchen_and_dining",
    "nutrition_info": "kitchen_and_dining",
    "recipe": "kitchen_and_dining",
    "restaurant_reviews": "kitchen_and_dining",
    "restaurant_reservation": "kitchen_and_dining",
    "meal_suggestion": "kitchen_and_dining",
    "restaurant_suggestion": "kitchen_and_dining",
    "cancel_reservation": "kitchen_and_dining",
    "ingredient_substitution": "kitchen_and_dining",
    "cook_time": "kitchen_and_dining",
    "accept_reservations": "kitchen_and_dining",
    # home
    "what_song": "home",
    "play_music": "home",
    "todo_list_update": "home",
    "reminder": "home",
    "reminder_update": "home",
    "calendar_update": "home",
    "order_status": "home",
    "update_playlist": "home",
    "shopping_list": "home",
    "calendar": "home",
    "next_song": "home",
    "order": "home",
    "todo_list": "home",
    "shopping_list_update": "home",
    "smart_home": "home",
    # auto_and_commute
    "current_location": "auto_and_commute",
    "oil_change_when": "auto_and_commute",
    "oil_change_how": "auto_and_commute",
    "uber": "auto_and_commute",
    "traffic": "auto_and_commute",
    "tire_pressure": "auto_and_commute",
    "schedule_maintenance": "auto_and_commute",
    "gas": "auto_and_commute",
    "mpg": "auto_and_commute",
    "distance": "auto_and_commute",
    "directions": "auto_and_commute",
    "last_maintenance": "auto_and_commute",
    "gas_type": "auto_and_commute",
    "tire_change": "auto_and_commute",
    "jump_start": "auto_and_commute",
    # travel
    "plug_type": "travel",
    "travel_notification": "travel",
    "translate": "travel",
    "flight_status": "travel",
    "international_visa": "travel",
    "timezone": "travel",
    "exchange_rate": "travel",
    "travel_suggestion": "travel",
    "travel_alert": "travel",
    "vaccines": "travel",
    "lost_luggage": "travel",
    "book_flight": "travel",
    "book_hotel": "travel",
    "carry_on": "travel",
    "car_rental": "travel",
    # utility
    "weather": "utility",
    "alarm": "utility",
    "date": "utility",
    "find_phone": "utility",
    "share_location": "utility",
    "timer": "utility",
    "make_call": "utility",
    "calculator": "utility",
    "definition": "utility",
    "measurement_conversion": "utility",
    "flip_coin": "utility",
    "spelling": "utility",
    "time": "utility",
    "roll_dice": "utility",
    "text": "utility",
    # work
    "pto_request_status": "work",
    "next_holiday": "work",
    "insurance_change": "work",
    "insurance": "work",
    "meeting_schedule": "work",
    "payday": "work",
    "taxes": "work",
    "income": "work",
    "rollover_401k": "work",
    "pto_balance": "work",
    "pto_request": "work",
    "w2": "work",
    "schedule_meeting": "work",
    "direct_deposit": "work",
    "pto_used": "work",
    # small_talk
    "who_made_you": "small_talk",
    "meaning_of_life": "small_talk",
    "who_do_you_work_for": "small_talk",
    "do_you_have_pets": "small_talk",
    "what_are_your_hobbies": "small_talk",
    "fun_fact": "small_talk",
    "what_is_your_name": "small_talk",
    "where_are_you_from": "small_talk",
    "goodbye": "small_talk",
    "thank_you": "small_talk",
    "greeting": "small_talk",
    "tell_joke": "small_talk",
    "are_you_a_bot": "small_talk",
    "how_old_are_you": "small_talk",
    "what_can_i_ask_you": "small_talk",
    # meta
    "change_speed": "meta",
    "user_name": "meta",
    "whisper_mode": "meta",
    "yes": "meta",
    "change_volume": "meta",
    "no": "meta",
    "change_language": "meta",
    "repeat": "meta",
    "change_accent": "meta",
    "cancel": "meta",
    "sync_device": "meta",
    "change_user_name": "meta",
    "change_ai_name": "meta",
    "reset_settings": "meta",
    "maybe": "meta",
}

CLINC150_DOMAIN_NAMES: tuple[str, ...] = (
    "banking",
    "credit_cards",
    "kitchen_and_dining",
    "home",
    "auto_and_commute",
    "travel",
    "utility",
    "work",
    "small_talk",
    "meta",
)

# ---------------------------------------------------------------------------
# Calibration and binning defaults
# ---------------------------------------------------------------------------

ECE_DEFAULT_BINS: int = 15
CONFIDENCE_BIN_EDGES: list[float] = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
CONFIDENCE_STRATA_LABELS: list[str] = [
    "[0.0, 0.2)",
    "[0.2, 0.4)",
    "[0.4, 0.6)",
    "[0.6, 0.8)",
    "[0.8, 1.0]",
]

# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------

HIGH_CONFIDENCE_THRESHOLD: float = 0.8
HIGH_CONFIDENCE_ERRORS_TOP_K: int = 20
WORST_CLASSES_K: int = 15
TOP_OOS_INTENTS_K: int = 10
TOP_OOS_EXAMPLES_K: int = 5
CONFUSION_STABILITY_TOP_K: int = 20

# ---------------------------------------------------------------------------
# Per-model analysis output filenames
# ---------------------------------------------------------------------------

EXTENDED_TEST_METRICS_FILENAME: str = "extended_test_metrics.json"
CALIBRATION_METRICS_FILENAME: str = "calibration_metrics.json"
RELIABILITY_DIAGRAM_FILENAME: str = "reliability_diagram.png"
CONFIDENCE_HISTOGRAM_FILENAME: str = "confidence_histogram.png"
OOS_THRESHOLD_METRICS_FILENAME: str = "oos_threshold_metrics.json"
OOS_ROC_CURVE_FILENAME: str = "oos_roc_curve.png"
OOS_PR_CURVE_FILENAME: str = "oos_pr_curve.png"
ERROR_TAXONOMY_FILENAME: str = "error_taxonomy.json"
OOS_ERROR_DEEP_DIVE_FILENAME: str = "oos_error_deep_dive.json"
OOS_ERROR_BREAKDOWN_FILENAME: str = "oos_error_breakdown.png"
CONFIDENCE_STRATIFICATION_FILENAME: str = "confidence_stratification.json"
CONFIDENCE_VS_ACCURACY_FILENAME: str = "confidence_vs_accuracy.png"
LENGTH_SLICE_FILENAME: str = "length_slice_analysis.json"
FREQUENCY_SLICE_FILENAME: str = "frequency_slice_analysis.json"
WORST_CLASSES_FILENAME: str = "worst_classes_deep_dive.json"
WORST_CLASSES_HEATMAP_FILENAME: str = "worst_classes_confusion_heatmap.png"
CONFUSION_STABILITY_FILENAME: str = "confusion_stability.json"
CONFUSION_STABILITY_FIGURE_FILENAME: str = "confusion_stability.png"

# ---------------------------------------------------------------------------
# Shared analysis output filenames
# ---------------------------------------------------------------------------

EXTENDED_METRICS_COMPARISON_FILENAME: str = "extended_metrics_comparison.json"
CALIBRATION_SUMMARY_FILENAME: str = "calibration_summary.json"
CALIBRATION_COMPARISON_FILENAME: str = "calibration_comparison.png"
OOS_THRESHOLD_COMPARISON_FILENAME: str = "oos_threshold_comparison.json"
OOS_ROC_COMPARISON_FILENAME: str = "oos_roc_comparison.png"
OOS_PR_COMPARISON_FILENAME: str = "oos_pr_comparison.png"
INTENT_DOMAIN_MAPPING_FILENAME: str = "intent_domain_mapping.json"
ERROR_TAXONOMY_SUMMARY_FILENAME: str = "error_taxonomy_summary.json"
ERROR_TAXONOMY_COMPARISON_FILENAME: str = "error_taxonomy_comparison.png"
OOS_FALSE_ACCEPT_COMPARISON_FILENAME: str = "oos_false_accept_comparison.json"
OOS_ERROR_COMPARISON_FILENAME: str = "oos_error_comparison.png"
CONFIDENCE_ACCURACY_COMPARISON_FILENAME: str = "confidence_accuracy_comparison.png"
LENGTH_SLICE_COMPARISON_FILENAME: str = "length_slice_comparison.png"
FREQUENCY_SLICE_COMPARISON_FILENAME: str = "frequency_slice_comparison.png"
CROSS_MODEL_ERROR_COMPARISON_FILENAME: str = "cross_model_error_comparison.json"
CROSS_MODEL_ERROR_OVERLAP_FILENAME: str = "cross_model_error_overlap.png"
UNIVERSALLY_MISCLASSIFIED_FILENAME: str = "universally_misclassified_examples.csv"
WORST_CLASSES_COMPARISON_FILENAME: str = "worst_classes_comparison.json"
CURATED_EXAMPLES_CSV_FILENAME: str = "curated_report_examples.csv"
CURATED_EXAMPLES_JSON_FILENAME: str = "curated_report_examples.json"
ERROR_ANALYSIS_SUMMARY_FILENAME: str = "error_analysis_summary.json"
ERROR_ANALYSIS_NOTES_FILENAME: str = "error_analysis_notes.md"
STEP11_HANDOFF_FILENAME: str = "step11_handoff.json"
