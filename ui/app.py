from __future__ import annotations

import json
import random
import time
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import streamlit as st

try:
    from components.model_a_tools import (
        load_metadata,
        load_sample_frame,
        load_question_ranker,
        load_unsupervised_results,
        build_option_frame,
        build_prediction_matrix,
        load_vectorizer,
        load_model,
        predict_option_probabilities,
        summarize_prediction,
    )
    from components.model_b_tools import (
        get_model_b_artifact_status,
        run_model_b_generation,
    )
except ImportError:  # pragma: no cover
    from ui.components.model_a_tools import (
        load_metadata,
        load_sample_frame,
        load_question_ranker,
        load_unsupervised_results,
        build_option_frame,
        build_prediction_matrix,
        load_vectorizer,
        load_model,
        predict_option_probabilities,
        summarize_prediction,
    )
    from ui.components.model_b_tools import (
        get_model_b_artifact_status,
        run_model_b_generation,
    )

try:
    from src.question_generation import generate_question
except ImportError:  # pragma: no cover
    from question_generation import generate_question

try:
    from src.model_a_generate import generate_question_details
except ImportError:  # pragma: no cover
    from model_a_generate import generate_question_details

try:
    from src.model_b_distractor import VOCAB_PATH, DISTRACTOR_RANKER_PATH, get_distractor_candidates
    from src.model_b_hint import HINT_SCORER_PATH, generate_hints
except ImportError:  # pragma: no cover
    from model_b_distractor import VOCAB_PATH, DISTRACTOR_RANKER_PATH, get_distractor_candidates
    from model_b_hint import HINT_SCORER_PATH, generate_hints


st.set_page_config(
    page_title="RACE RC Quiz System - Model A + Model B",
    page_icon="📘",
    layout="wide",
    initial_sidebar_state="expanded",
)

ROOT_DIR = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT_DIR / "models" / "model_a" / "traditional"
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
DATA_DIR = ROOT_DIR / "data" / "raw"
MODEL_A_RESULTS_PATH = MODELS_DIR / "evaluation_results.json"
MODEL_B_DIR = ROOT_DIR / "models" / "model_b"
MODEL_B_EVAL_PATH = MODEL_B_DIR / "evaluation_results.json"
EXTERNAL_RESULTS_PATH = ROOT_DIR.parent / "results_summary.json"


def apply_consistent_theme() -> None:
    """Inject lightweight styling so both model sections feel visually unified."""
    st.markdown(
        """
        <style>
        .stApp {
            background: linear-gradient(135deg, #0f1724 0%, #0b1220 55%, #02040a 100%);
            color: #e6eef8;
        }
        .stApp .css-1d391kg { color: #e6eef8; }
        .hero {
            padding: 1.15rem 1.3rem;
            border: 1px solid rgba(255,255,255,0.04);
            border-radius: 1rem;
            background: rgba(10,18,28,0.85);
            box-shadow: 0 8px 24px rgba(0,0,0,0.6);
        }
        .pill-row { display:flex; flex-wrap:wrap; gap:0.5rem; margin-top:0.55rem; }
        .pill {
            padding:0.3rem 0.72rem; border-radius:999px; border:1px solid rgba(99,102,241,0.18);
            background:rgba(99,102,241,0.08); color:#f8fafc; font-size:0.82rem; font-weight:600;
        }
        .panel-card {
            border:1px solid rgba(255,255,255,0.04); background:rgba(6,10,18,0.8);
            border-radius:0.95rem; padding:1rem; color:#e6eef8;
        }
        .panel-note {
            border-left: 4px solid rgba(99,102,241,0.6);
            padding: 0.45rem 0.7rem;
            background: rgba(99,102,241,0.04);
            border-radius: 0.35rem;
            margin-bottom: 0.75rem;
            color: #e6eef8;
        }
        mark.quiz-highlight { background:#ffb86b; color:#0b1220; padding:0.05rem 0.15rem; border-radius:0.2rem; }
        .stMarkdown, .stText, .stButton > button { color: #e6eef8; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def apply_theme() -> None:
    """Backward-compatible alias used by the alternate `main` entrypoint."""
    apply_consistent_theme()


def _safe_json_load(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


@st.cache_data(show_spinner=False)
def get_samples(split: str) -> pd.DataFrame:
    try:
        frame = load_sample_frame(split)
        return frame.sample(frac=1.0, random_state=42).reset_index(drop=True)
    except Exception:
        return pd.DataFrame()


@st.cache_data(show_spinner=False)
def get_sample_count(split: str) -> int:
    return len(get_samples(split))


def get_sample_preview(split: str, sample_index: int) -> dict[str, Any]:
    frame = get_samples(split)
    if frame.empty:
        raise ValueError(f"No samples available for split '{split}'.")

    sample = frame.iloc[sample_index % len(frame)]
    options = {label: str(sample.get(label, "")) for label in ["A", "B", "C", "D"]}

    return {
        "article": str(sample.get("article", "")),
        "question": str(sample.get("question", "")),
        "options": options,
        "answer": str(sample.get("answer", "")).strip().upper(),
        "id": str(sample.get("id", "")),
    }


@st.cache_resource(show_spinner=False)
def get_question_generation_artifacts():
    """Load the vectorizer and ranker used by the generation tab."""
    vectorizer = load_vectorizer()
    try:
        ranker = load_question_ranker()
    except Exception:
        ranker = None
    return vectorizer, ranker


def run_prediction(model_name: str, article: str, question: str, options: dict[str, str]):
    result = predict_option_probabilities(model_name, article, question, options)
    return summarize_prediction(result), result


def prediction_panel(model_name: str, article: str, question: str, options: dict[str, str], known_answer: str | None):
    summary, result = run_prediction(model_name, article, question, options)

    top_option = summary["predicted_option"]
    confidence = summary["confidence"]

    left, right = st.columns([1.2, 0.8], gap="large")
    with left:
        st.subheader("Ranked options")
        for row in result.itertuples(index=False):
            with st.container(border=True):
                col_a, col_b = st.columns([0.8, 0.2])
                with col_a:
                    st.markdown(f"**Option {row.option_label}**")
                with col_b:
                    st.markdown(f"**{row.prob_correct:.3f}**")

                option_text = row.option_text if row.option_text else "*(empty option)*"
                st.write(option_text)

    with right:
        st.subheader("Model verdict")
        st.metric(label="Predicted option", value=top_option, delta=summary.get("predicted_text", ""))
        st.metric(label="Confidence", value=f"{confidence:.3f}", help="Predicted probability for the top option")

        st.divider()

        if known_answer:
            verdict = "Correct" if top_option == known_answer else "Incorrect"
            if verdict == "Correct":
                st.success(f"**Compared with ground truth: {verdict}**\n\nGround truth was {known_answer}.")
            else:
                st.error(f"**Compared with ground truth: {verdict}**\n\nGround truth was {known_answer}.")

        st.info(
            "Model A is an answer-verification model. It ranks the four options and predicts which one is most likely correct.",
            icon="ℹ️",
        )


def custom_test_tab() -> None:
    st.header("1) Custom Model A test")
    st.write(
        "Paste a passage, question, and four answer options. If you know the ground-truth answer, choose it to check whether Model A got it right."
    )

    with st.form("custom_test_form"):
        col1, col2 = st.columns([1.2, 0.8], gap="large")

        with col1:
            article = st.text_area(
                "Article / passage",
                height=300,
                placeholder="Paste the reading passage here...",
            )
            question = st.text_input("Question", placeholder="What does the passage mainly say?")
            st.caption("💡 Tip: Use the Sample Browser tab to copy a real passage and test it here.")

        with col2:
            st.subheader("Options")
            option_a = st.text_input("Option A", placeholder="First answer choice")
            option_b = st.text_input("Option B", placeholder="Second answer choice")
            option_c = st.text_input("Option C", placeholder="Third answer choice")
            option_d = st.text_input("Option D", placeholder="Fourth answer choice")

            st.divider()
            known_answer = st.radio("Ground truth answer (if known)", ["", "A", "B", "C", "D"], horizontal=True)
            model_name = st.selectbox(
                "Model to test",
                ["random_forest", "logistic_regression", "ensemble"],
                index=0,
                help="Random Forest is the best single-model default; the ensemble uses the tuned threshold.",
            )

        submit_button = st.form_submit_button("Run Model A", type="primary", use_container_width=True)

    if submit_button:
        if not article.strip() or not question.strip() or not all([option_a.strip(), option_b.strip(), option_c.strip(), option_d.strip()]):
            st.error("Please fill in the article, question, and all four options before running Model A.")
        else:
            with st.spinner("Scoring options with Model A..."):
                prediction_panel(
                    model_name=model_name,
                    article=article,
                    question=question,
                    options={"A": option_a, "B": option_b, "C": option_c, "D": option_d},
                    known_answer=known_answer if known_answer else None,
                )


@st.cache_data(show_spinner=False)
def get_metrics_table() -> pd.DataFrame:
    try:
        evaluation_path = MODELS_DIR / "evaluation_results.json"
        if evaluation_path.exists():
            payload = json.loads(evaluation_path.read_text(encoding="utf-8"))
            rows = []
            for model_name, metrics in payload.items():
                rows.append(
                    {
                        "Model": str(model_name).replace("_", " ").title(),
                        "Accuracy": metrics.get("accuracy"),
                        "Balanced Accuracy": metrics.get("balanced_accuracy"),
                        "Macro F1": metrics.get("macro_f1"),
                        "Exact Match": metrics.get("exact_match"),
                        "Recall (Class 1)": metrics.get("recall_class_1"),
                    }
                )
        else:
            metadata = load_metadata()
            metrics = metadata.get("metrics_test", {})
            rows = [
                {
                    "Model": model_name.replace("_", " ").title(),
                    "Accuracy": data.get("accuracy"),
                    "Balanced Accuracy": data.get("balanced_accuracy"),
                    "Macro F1": data.get("f1_macro"),
                    "Recall (Class 1)": data.get("recall_class_1"),
                }
                for model_name, data in metrics.items()
            ]
        return pd.DataFrame(rows)
    except Exception:
        return pd.DataFrame()


@st.cache_data(show_spinner=False)
def get_unsupervised_result(name: str) -> dict[str, Any]:
    try:
        return load_unsupervised_results(name)
    except Exception:
        return {}


def _metric_card(label: str, value: str, delta: str) -> None:
    with st.container(border=True):
        st.metric(label, value, delta)


def _render_image_panel(title: str, image_path: Path, caption: str, missing_message: str) -> None:
    """Render an image if available and explain when it cannot be shown."""
    st.subheader(title)
    if image_path.exists():
        try:
            st.image(image_path, caption=caption, use_container_width=True)
            st.caption(f"Loaded from {image_path.name}")
        except Exception as exc:
            st.warning(f"Found {image_path.name}, but Streamlit could not render it: {exc}")
    else:
        st.info(missing_message)


def _show_metric_definitions() -> None:
    """Explain what the displayed scores mean."""
    with st.expander("What the metrics mean", expanded=False):
        st.markdown(
            """
            - **Accuracy / Exact Match**: fraction of questions where the top-ranked option is the true answer.
            - **Balanced Accuracy**: average recall across classes; useful when one class is more common.
            - **Macro F1**: mean F1 across classes, treating both classes equally.
            - **Recall (Class 1)**: how often the model finds the correct answer option.
            - **Silhouette**: cluster separation score in unsupervised learning; higher means cleaner grouping.
            - **Davies-Bouldin**: cluster overlap score; lower means better separation.
            - **BIC / AIC**: lower values generally indicate a better fit with less complexity.
            - **Soft confidence**: average max cluster probability in GMM; higher means the model is more certain.
            """
        )


def unsupervised_tab() -> None:
    st.header("5) Unsupervised and semi-supervised results")
    st.write(
        "This tab shows the actual experiment outputs from K-Means and semi-supervised learning, including how they compare with the supervised baselines."
    )

    kmeans = get_unsupervised_result("kmeans")
    gmm = get_unsupervised_result("gmm")
    label_prop = get_unsupervised_result("label_propagation")

    if not kmeans and not gmm and not label_prop:
        st.warning("No unsupervised artifacts were found in models/model_a/unsupervised/.")
        return

    _show_metric_definitions()

    cols = st.columns(4)
    with cols[0]:
        _metric_card(
            "K-Means silhouette",
            f"{kmeans.get('train_results', {}).get('silhouette_score', 0.0):.3f}",
            "Higher means tighter clusters",
        )
    with cols[1]:
        _metric_card(
            "GMM silhouette",
            f"{gmm.get('results', {}).get('silhouette', 0.0):.3f}",
            "Higher means better separation",
        )
    with cols[2]:
        _metric_card(
            "K-Means separation",
            f"{kmeans.get('train_results', {}).get('cluster_separation', 0.0):.3f}",
            "0 = no split, 1 = perfect split",
        )
    with cols[3]:
        _metric_card(
            "GMM confidence",
            f"{gmm.get('results', {}).get('soft_label_mean_confidence', 0.0):.3f}",
            "Average strongest cluster probability",
        )

    second_row = st.columns(2)
    with second_row[0]:
        _metric_card(
            "Label Propagation accuracy",
            f"{label_prop.get('label_propagation', {}).get('accuracy', 0.0):.3f}",
            "Top-label match rate on the test set",
        )
    with second_row[1]:
        _metric_card(
            "Self-Training accuracy",
            f"{label_prop.get('self_training', {}).get('accuracy', 0.0):.3f}",
            "Top-label match rate on the test set",
        )

    st.subheader("Comparison table")
    comparison_rows = []
    if kmeans:
        train_results = kmeans.get("train_results", {})
        comparison_rows.append(
            {
                "Method": "K-Means",
                "Silhouette": train_results.get("silhouette_score"),
                "Davies-Bouldin": train_results.get("davies_bouldin_index"),
                "Cluster separation": train_results.get("cluster_separation"),
            }
        )
    if gmm:
        gmm_results = gmm.get("results", {})
        comparison_rows.append(
            {
                "Method": "GMM",
                "Silhouette": gmm_results.get("silhouette"),
                "BIC": gmm_results.get("bic"),
                "AIC": gmm_results.get("aic"),
                "Soft confidence": gmm_results.get("soft_label_mean_confidence"),
            }
        )
    if label_prop:
        lp = label_prop.get("label_propagation", {})
        st_data = label_prop.get("self_training", {})
        comparison_rows.extend(
            [
                {
                    "Method": "Label Propagation",
                    "Accuracy": lp.get("accuracy"),
                    "Weighted F1": lp.get("weighted_f1"),
                    "Recall class 1": (lp.get("recall_per_class") or [None, None])[1],
                },
                {
                    "Method": "Self-Training",
                    "Accuracy": st_data.get("accuracy"),
                    "Weighted F1": st_data.get("weighted_f1"),
                    "Recall class 1": (st_data.get("recall_per_class") or [None, None])[1],
                },
            ]
        )

    if comparison_rows:
        st.dataframe(pd.DataFrame(comparison_rows), use_container_width=True, hide_index=True)

    st.info(
        "The numbers above are different kinds of scores: supervised metrics measure how often the model picks the correct option, while unsupervised metrics measure whether the clusters look separated and confident."
    )

    col_left, col_right = st.columns(2, gap="large")
    with col_left:
        image_path = ROOT_DIR / "models" / "model_a" / "unsupervised" / "kmeans_pca_visualization.png"
        _render_image_panel(
            "K-Means visualization",
            image_path,
            "K-Means clusters in 2D TruncatedSVD space",
            "K-Means visualization not found yet.",
        )

    with col_right:
        image_path = ROOT_DIR / "models" / "model_a" / "unsupervised" / "gmm_pca_visualization.png"
        _render_image_panel(
            "GMM visualization",
            image_path,
            "GMM clusters and soft assignment confidence",
            "GMM visualization not found yet.",
        )

    image_path = ROOT_DIR / "models" / "model_a" / "unsupervised" / "label_propagation_comparison.png"
    _render_image_panel(
        "Semi-supervised comparison",
        image_path,
        "Semi-supervised vs supervised comparison",
        "Label propagation comparison image not found yet.",
    )

    if kmeans:
        with st.expander("K-Means details", expanded=False):
            st.write(kmeans.get("interpretation", {}))
            st.json({
                "model": kmeans.get("model"),
                "task": kmeans.get("task"),
                "train_results": kmeans.get("train_results", {}),
            })

    if label_prop:
        with st.expander("Semi-supervised details", expanded=False):
            st.write(label_prop.get("interpretation", {}))
            st.json({
                "label_propagation": {
                    "accuracy": label_prop.get("label_propagation", {}).get("accuracy"),
                    "weighted_f1": label_prop.get("label_propagation", {}).get("weighted_f1"),
                    "recall_per_class": label_prop.get("label_propagation", {}).get("recall_per_class"),
                },
                "self_training": {
                    "accuracy": label_prop.get("self_training", {}).get("accuracy"),
                    "weighted_f1": label_prop.get("self_training", {}).get("weighted_f1"),
                    "recall_per_class": label_prop.get("self_training", {}).get("recall_per_class"),
                },
                "baselines": label_prop.get("baselines", {}),
            })


def sample_browser_tab() -> None:
    st.header("2) Sample browser")
    st.write("Use a real RACE sample from the corrected dataset to test Model A quickly.")

    split = st.selectbox("Dataset split", ["test", "val"], index=0)
    sample_count = get_sample_count(split)

    if sample_count == 0:
        st.warning(f"No samples found in the {split} split.")
        return

    index = st.slider("Sample index", 0, sample_count - 1, 0)
    sample = get_sample_preview(split, index)
    st.session_state["sample_browser_sample"] = sample

    with st.container(border=True):
        st.caption(f"Sample {index + 1} from {split.upper()} split")
        st.subheader(sample["question"])

    with st.container(border=True):
        st.write(sample["article"])

    options = sample["options"]
    true_answer = sample["answer"]

    with st.expander("Show answer choices", expanded=True):
        for label in ["A", "B", "C", "D"]:
            option_text = options[label] or "*(empty)*"
            st.write(f"**{label}.** {option_text}")

    st.caption(f"**Sample ID:** {sample['id']} | **Ground truth answer:** {true_answer or 'unknown'}")

    if st.button("Test this sample with Model A", type="primary", use_container_width=True):
        with st.spinner("Scoring sample..."):
            prediction_panel(
                model_name="ensemble",
                article=sample["article"],
                question=sample["question"],
                options=options,
                known_answer=true_answer,
            )


def model_card_tab() -> None:
    st.header("3) Model summary")
    st.write("This tab shows the deployed Model A setup and the most important metrics.")

    metrics_table = get_metrics_table()
    if not metrics_table.empty:
        st.dataframe(metrics_table, use_container_width=True, hide_index=True)
        st.caption(
            "Accuracy / Exact Match are the fraction of questions where the top-ranked answer is correct. Balanced Accuracy and Macro F1 are more informative when the classes are imbalanced."
        )

    try:
        metadata = load_metadata()
        threshold = metadata.get("selected_models", {}).get("ensemble_threshold", 0.49)
    except Exception:
        threshold = 0.49

    cols = st.columns(3)
    with cols[0]:
        with st.container(border=True):
            st.metric("Default single model", "Random Forest", "Best overall score")
    with cols[1]:
        with st.container(border=True):
            st.metric("Backup model", "Logistic Regression", "Higher recall on correct answers")
    with cols[2]:
        with st.container(border=True):
            st.metric("Ensemble threshold", f"{threshold:.2f}", "Averaged positive-class probability")

    st.subheader("Files in the deployment package")
    st.code(
        """
models/model_a/traditional/
├── model_a_random_forest.pkl
├── model_a_logistic_regression.pkl
├── model_a_svm_calibrated.pkl
├── model_a_ensemble.pkl
├── question_ranker.pkl
├── model_metadata.json
├── evaluation_results.json
└── model_card.md
""".strip(),
        language="text",
    )

    st.subheader("How to run locally")
    st.code("python3 -m streamlit run ui/app.py", language="bash")


def how_it_works_tab() -> None:
    st.header("4) How Model A works")
    st.write(
        "Model A takes a passage, question, and four candidate answers, vectorizes the combined text, and scores each option. The UI then ranks the options and highlights the most likely correct answer."
    )

    st.markdown(
        """
        - **Single-model mode**: use Random Forest or Logistic Regression individually.
        - **Ensemble mode**: calibrated soft voting across Random Forest, Logistic Regression, and Linear SVC.
        - **Decision threshold**: the ensemble uses the stored threshold of 0.49 for binary decisions.
        - **Best use case**: test whether the model can identify the correct answer from four candidates.
        """
    )

    st.info(
        "If you want to verify that the deployment package is working, use the sample browser tab first. It reads from the corrected RACE split and uses the same vectorizer/model artifacts as inference."
    )


def question_generation_tab() -> None:
    """Render the question generation tab."""
    st.header("6) Generate Question")
    st.write("Paste an article and the known answer text to generate a template-based question, then optionally verify the four answer options.")

    vectorizer, ranker = get_question_generation_artifacts()
    sample = st.session_state.get("sample_browser_sample", {})

    if "generated_question_text" not in st.session_state:
        st.session_state["generated_question_text"] = ""
    if "generated_question_source" not in st.session_state:
        st.session_state["generated_question_source"] = ""
    if "generated_question_wh" not in st.session_state:
        st.session_state["generated_question_wh"] = "what"

    use_sample = st.checkbox("Prefill from the latest sample browser item", value=bool(sample))
    default_article = sample.get("article", "") if use_sample else ""
    default_answer = ""
    default_options = {"A": "", "B": "", "C": "", "D": ""}
    if use_sample:
        default_options = sample.get("options", default_options)
        answer_label = sample.get("answer", "")
        if answer_label in default_options:
            default_answer = default_options.get(answer_label, "")

    with st.form("question_generation_form"):
        article = st.text_area("Article", value=default_article, height=260, placeholder="Paste the passage here...")
        correct_answer = st.text_input("Correct answer text", value=default_answer, placeholder="Type the known answer here...")
        submit = st.form_submit_button("Generate Question", type="primary", use_container_width=True)

    if submit:
        if not article.strip() or not correct_answer.strip():
            st.error("Please provide both the article and the correct answer text.")
        else:
            with st.spinner("Generating question..."):
                generated_question = generate_question(article, correct_answer, vectorizer, ranker_model=ranker)
                st.session_state["generated_question_text"] = generated_question["question"]
                st.session_state["generated_question_source"] = generated_question["source_sentence"]
                st.session_state["generated_question_wh"] = generated_question["wh_word"]

    if st.session_state["generated_question_text"]:
        st.success(st.session_state["generated_question_text"])
        st.caption(f"Source sentence: {st.session_state['generated_question_source'] or 'No source sentence identified.'}")
        st.caption(f"WH-word: {st.session_state['generated_question_wh']}")
        st.caption("You can edit the generated question before verifying the answer options.")
        edited_question = st.text_area(
            "Generated question",
            value=st.session_state["generated_question_text"],
            height=120,
            key="generated_question_editor",
        )
        st.session_state["generated_question_text"] = edited_question.strip()
    else:
        edited_question = ""

    st.divider()
    st.subheader("Verify answer options")
    st.write("Type four options manually or reuse the latest sample browser options.")

    with st.form("question_verifier_form"):
        option_a = st.text_input("Option A", value=default_options.get("A", ""))
        option_b = st.text_input("Option B", value=default_options.get("B", ""))
        option_c = st.text_input("Option C", value=default_options.get("C", ""))
        option_d = st.text_input("Option D", value=default_options.get("D", ""))
        verify = st.form_submit_button("Run answer verifier", type="primary", use_container_width=True)

    if verify:
        if not article.strip() or not correct_answer.strip():
            st.error("Generate the question first or keep the article and answer fields populated.")
        elif not all([option_a.strip(), option_b.strip(), option_c.strip(), option_d.strip()]):
            st.error("Please fill in all four options before running the verifier.")
        else:
            question_text = st.session_state.get("generated_question_text", "").strip()
            if not question_text:
                generated_question = generate_question(article, correct_answer, vectorizer, ranker_model=ranker)
                question_text = generated_question["question"]
                st.session_state["generated_question_text"] = question_text

            prediction_panel(
                model_name="ensemble",
                article=article,
                question=question_text,
                options={"A": option_a, "B": option_b, "C": option_c, "D": option_d},
                known_answer=None,
            )


def model_b_tab() -> None:
    """Render the Model B middle-ground distractor + hint pipeline."""
    st.header("7) Model B pipeline")
    st.write(
        "Model B generates three distractors and three graduated hints from the passage, question, and correct answer text."
    )
    st.markdown(
        "<div class='panel-note'>Model B is locked to the middle-ground path: Word2Vec + one-hot retrieval for distractors, plus the trained hint scorer.</div>",
        unsafe_allow_html=True,
    )

    status = get_model_b_artifact_status()
    if status["missing"]:
        st.error("Model B artifacts are missing. Please place the files below in models/model_b/traditional/.")
        st.code("\n".join(status["missing"]), language="text")
        return

    st.caption(f"Artifacts loaded from: {status['model_dir']}")

    sample = st.session_state.get("sample_browser_sample", {})
    use_sample = st.checkbox("Prefill from the latest sample browser item", value=bool(sample), key="model_b_prefill")

    default_article = sample.get("article", "") if use_sample else ""
    default_question = sample.get("question", "") if use_sample else ""
    default_answer = ""
    if use_sample:
        options = sample.get("options", {})
        answer_label = sample.get("answer", "")
        if answer_label in options:
            default_answer = options.get(answer_label, "")

    with st.form("model_b_form"):
        article = st.text_area("Article", value=default_article, height=240, placeholder="Paste article text...")
        question = st.text_input("Question", value=default_question, placeholder="Enter the question...")
        answer_text = st.text_input("Correct answer text", value=default_answer, placeholder="Enter the known correct answer...")
        run_model_b = st.form_submit_button("Run Model B", type="primary", use_container_width=True)

    if not run_model_b:
        return

    if not article.strip() or not question.strip() or not answer_text.strip():
        st.error("Please provide article, question, and correct answer text.")
        return

    with st.spinner("Running middle-ground Model B generation..."):
        result = run_model_b_generation(article=article, question=question, answer_text=answer_text)

    diagnostics = result.get("diagnostics", {})
    metric_cols = st.columns(3)
    with metric_cols[0]:
        st.metric("Generated distractors", f"{len(result.get('distractors', []))}")
    with metric_cols[1]:
        st.metric("Diversity", f"{diagnostics.get('distractor_diversity', 0.0):.3f}")
    with metric_cols[2]:
        st.metric("Max answer overlap", f"{diagnostics.get('answer_overlap_max', 0.0):.3f}")

    col_left, col_right = st.columns(2, gap="large")
    with col_left:
        st.subheader("Generated options")
        for label in ["A", "B", "C", "D"]:
            option_text = result["options"].get(label, "")
            with st.container(border=True):
                st.markdown(f"**{label}.** {option_text if option_text else '*(empty)*'}")
                if label == result.get("correct_option"):
                    st.success("Correct option")

    with col_right:
        st.subheader("Distractors")
        for idx, distractor in enumerate(result.get("distractors", []), start=1):
            with st.container(border=True):
                st.markdown(f"**Distractor {idx}**")
                st.write(distractor)

    st.subheader("Graduated hints")
    for idx, hint in enumerate(result.get("hints", []), start=1):
        with st.container(border=True):
            st.markdown(f"**Hint {idx}**")
            st.write(hint)

    st.info(
        "Model B is a support-generation pipeline: it proposes plausible wrong options and hint sentences to improve quiz quality and learner guidance.",
        icon="ℹ️",
    )


def render_hero() -> None:
    try:
        metadata = load_metadata()
        threshold = metadata.get("selected_models", {}).get("ensemble_threshold", 0.49)
    except Exception:
        threshold = 0.49

    st.title("RACE Reading Comprehension Quiz System")
    st.markdown(
        "<div class='pill-row'><span class='pill'>Model A: Answer Verification</span><span class='pill'>Model B: Distractors + Hints</span><span class='pill'>Middle-ground Model B</span></div>",
        unsafe_allow_html=True,
    )
    st.write("Use one unified interface to verify answers with Model A and generate distractors and hints with Model B.")
    st.info(f"Model A ensemble threshold: **{threshold:.2f}** | Default single model: **Random Forest**", icon="🧠")


def main() -> None:
    apply_consistent_theme()
    render_hero()
    st.write("")

    with st.sidebar:
        st.header("Controls")
        st.caption("Model A verifies answer options. Model B generates distractors and hints.")

        st.subheader("Quick links")
        st.markdown("- 🧪 Test a custom question")
        st.markdown("- 📚 Browse a real sample")
        st.markdown("- 📊 Inspect model metrics")
        st.markdown("- 📖 Read how the model works")
        st.markdown("- 🧩 Test middle-ground Model B")

        st.subheader("Deployment files")
        st.code("ui/app.py\nui/components/model_a_tools.py\nui/components/model_b_tools.py", language="text")

    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(
        [
            "Test Model A",
            "Sample browser",
            "Model summary",
            "How it works",
            "Unsupervised results",
            "Generate Question",
            "Model B",
        ]
    )

    with tab1:
        custom_test_tab()
    with tab2:
        sample_browser_tab()
    with tab3:
        model_card_tab()
    with tab4:
        how_it_works_tab()
    with tab5:
        unsupervised_tab()
    with tab6:
        question_generation_tab()
    with tab7:
        model_b_tab()


if __name__ == "__main__":
    main()
def load_race_samples() -> pd.DataFrame:
    sample_path = DATA_DIR / "test.csv"
    if not sample_path.exists():
        return pd.DataFrame()
    frame = pd.read_csv(sample_path)
    required = {"article", "question", "A", "B", "C", "D", "answer"}
    return frame if required.issubset(frame.columns) else pd.DataFrame()


def _sample_answer_text(row: pd.Series) -> str:
    answer_label = str(row.get("answer", "")).strip().upper()
    return str(row.get(answer_label, "")).strip() if answer_label in {"A", "B", "C", "D"} else ""


def _pick_random_sample() -> dict[str, str]:
    frame = load_race_samples()
    if frame.empty:
        raise ValueError("No RACE samples are available.")
    row = frame.sample(n=1, random_state=random.randint(0, 10_000)).iloc[0]
    return {
        "article": str(row.get("article", "")),
        "question": str(row.get("question", "")),
        "correct_answer": _sample_answer_text(row),
        "sample_id": str(row.get("id", "")),
    }


def _ensure_state_defaults() -> None:
    defaults = {
        "article_input": "",
        "question_input": "",
        "correct_answer_input": "",
        "model_b_mode": "middle_ground",
        "current_quiz": None,
        "quiz_version": 0,
        "session_log": [],
        "hint_used_1": False,
        "hint_used_2": False,
        "hint_used_3": False,
        "selected_option": None,
        "last_inference": {},
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def _clean_text(value: object) -> str:
    if value is None:
        return ""
    text = str(value).lower()
    for symbol in "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~":
        text = text.replace(symbol, " ")
    return " ".join(text.split())


def _highlight_sentence(article: str, sentence: str) -> str:
    article_clean = article.replace("\n", " ")
    sentence_clean = sentence.replace("\n", " ").strip()
    if not sentence_clean:
        return article_clean
    return article_clean.replace(sentence_clean, f"<mark class='quiz-highlight'>{sentence_clean}</mark>")


def _fallback_distractors(article: str, correct_answer: str, vocab: list[str], count: int = 3) -> list[str]:
    article_tokens = [token for token in _clean_text(article).split() if token not in _clean_text(correct_answer)]
    answer_clean = _clean_text(correct_answer)
    candidates: list[str] = []
    for token in article_tokens:
        if token and token not in candidates and token not in answer_clean:
            candidates.append(token)
        if len(candidates) >= count:
            break
    if len(candidates) < count:
        for token in vocab:
            if token not in article_tokens and token not in answer_clean and token not in candidates:
                candidates.append(token)
            if len(candidates) >= count:
                break
    while len(candidates) < count:
        candidates.append("None of the above")
    return candidates[:count]


def _fallback_hints(article: str, correct_answer: str) -> list[str]:
    sentences = [sentence.strip() for sentence in article.replace("\n", " ").split(". ") if len(sentence.split()) >= 5]
    if not sentences:
        return ["Read the passage carefully.", "Look for the key detail in the passage.", "The answer is stated in the passage."]
    top = sentences[0]
    second = sentences[1] if len(sentences) > 1 else top
    return [
        top.replace(correct_answer, "[...]") if correct_answer else top,
        second.replace(correct_answer, f"{correct_answer[: max(1, len(correct_answer) // 2)]}[...]") if correct_answer else second,
        f"The answer can be found in this part: '{top[:80]}...'.",
    ]


def _quiz_sentence(article: str, generated_question: str) -> str:
    sentences = [sentence.strip() for sentence in article.replace("\n", " ").split(". ") if sentence.strip()]
    if not sentences:
        return ""
    q_tokens = set(_clean_text(generated_question).split())
    best_sentence = sentences[0]
    best_overlap = -1
    for sentence in sentences:
        overlap = len(set(_clean_text(sentence).split()) & q_tokens)
        if overlap > best_overlap:
            best_overlap = overlap
            best_sentence = sentence
    return best_sentence


def _model_b_mode_label(mode: str) -> str:
    if mode == "one_hot":
        return "One-hot only"
    return "Middle ground (Word2Vec + one-hot)"


@st.cache_resource(show_spinner=False)
def load_models() -> dict[str, Any]:
    vectorizer = load_vectorizer()
    verification_model = load_model("ensemble")
    try:
        question_ranker = load_question_ranker()
    except Exception:
        question_ranker = None

    vocab_payload: dict[str, Any] = {}
    if VOCAB_PATH.exists():
        try:
            vocab_payload = joblib.load(VOCAB_PATH)
        except Exception:
            vocab_payload = {}

    distractor_ranker = None
    if DISTRACTOR_RANKER_PATH.exists():
        try:
            distractor_ranker = joblib.load(DISTRACTOR_RANKER_PATH)
        except Exception:
            distractor_ranker = None

    hint_scorer = None
    if HINT_SCORER_PATH.exists():
        try:
            hint_scorer = joblib.load(HINT_SCORER_PATH)
        except Exception:
            hint_scorer = None

    try:
        w2v_model = load_word2vec_model()
    except Exception:
        w2v_model = None

    return {
        "vectorizer": vectorizer,
        "verification_model": verification_model,
        "question_ranker": question_ranker,
        "vocab": vocab_payload.get("vocab", []),
        "word2idx": vocab_payload.get("word2idx", {}),
        "distractor_ranker": distractor_ranker,
        "hint_scorer": hint_scorer,
        "w2v_model": w2v_model,
        "model_b_eval": _safe_json_load(MODEL_B_EVAL_PATH),
        "model_a_eval": _safe_json_load(MODEL_A_RESULTS_PATH),
        "summary": _safe_json_load(EXTERNAL_RESULTS_PATH),
    }


def _build_quiz(article: str, correct_answer: str, question_seed: str, models: dict[str, Any], model_b_mode: str) -> dict[str, Any]:
    timings: dict[str, float] = {}

    start = time.perf_counter()
    question_details = generate_question_details(article, correct_answer, models.get("question_ranker"))
    timings["question_generation"] = time.perf_counter() - start

    generated_question = question_details.get("question") or question_seed or "What does the passage say?"
    source_sentence = question_details.get("source_sentence", "") or _quiz_sentence(article, generated_question)

    start = time.perf_counter()
    if (
        model_b_mode == "middle_ground"
        and models.get("distractor_ranker")
        and models.get("vocab")
        and models.get("word2idx")
        and models.get("w2v_model") is not None
    ):
        distractors = generate_distractors_combined(
            article=article,
            question=generated_question,
            correct_answer=correct_answer,
            vocab=models["vocab"],
            word2idx=models["word2idx"],
            ranker_model=models["distractor_ranker"],
            w2v_model=models["w2v_model"],
            n=3,
        )
    elif models.get("distractor_ranker") and models.get("vocab") and models.get("word2idx"):
        distractors = get_distractor_candidates(article, correct_answer, models["vocab"], models["word2idx"], top_n=3)
    else:
        distractors = _fallback_distractors(article, correct_answer, models.get("vocab", []), count=3)
    timings["distractor_generation"] = time.perf_counter() - start

    start = time.perf_counter()
    if models.get("hint_scorer") is not None:
        hints = generate_hints(article, generated_question, correct_answer, models["hint_scorer"])
    else:
        hints = _fallback_hints(article, correct_answer)
    timings["hint_generation"] = time.perf_counter() - start

    options = [correct_answer] + [item for item in distractors if item and _clean_text(item) != _clean_text(correct_answer)]
    options = list(dict.fromkeys(options))[:4]
    while len(options) < 4:
        options.append(f"Option {len(options) + 1}")
    random.shuffle(options)
    labels = ["A", "B", "C", "D"]
    option_map = {label: options[index] for index, label in enumerate(labels)}
    correct_label = next((label for label, value in option_map.items() if _clean_text(value) == _clean_text(correct_answer)), "")

    return {
        "generated_question": generated_question,
        "source_sentence": source_sentence,
        "options": option_map,
        "correct_label": correct_label,
        "correct_answer": correct_answer,
        "distractors": distractors,
        "hints": hints,
        "timings": timings,
        "original_question": question_seed,
    }


def _score_options(article: str, question: str, options: dict[str, str], models: dict[str, Any]) -> pd.DataFrame:
    option_frame = build_option_frame(article, question, options)
    matrix = build_prediction_matrix(option_frame, models["vectorizer"])
    proba = models["verification_model"].predict_proba(matrix)[:, 1]
    scored = option_frame.copy()
    scored["prob_correct"] = proba
    return scored.sort_values("prob_correct", ascending=False).reset_index(drop=True)


def _store_session_result(quiz: dict[str, Any], selected_label: str, predicted_label: str, verifier_correct: bool, selected_correct: bool, verification_time: float) -> None:
    st.session_state.session_log.append(
        {
            "timestamp": pd.Timestamp.now().isoformat(),
            "generated_question": quiz["generated_question"],
            "original_question": quiz.get("original_question", ""),
            "correct_answer": quiz["correct_answer"],
            "selected_label": selected_label,
            "selected_text": quiz["options"].get(selected_label, ""),
            "predicted_label": predicted_label,
            "predicted_text": quiz["options"].get(predicted_label, ""),
            "selected_correct": bool(selected_correct),
            "verifier_correct": bool(verifier_correct),
            "model_a_latency": float(quiz["timings"].get("question_generation", 0.0)),
            "model_b_latency": float(quiz["timings"].get("distractor_generation", 0.0) + quiz["timings"].get("hint_generation", 0.0)),
            "verification_latency": float(verification_time),
            "source_sentence": quiz.get("source_sentence", ""),
        }
    )


def render_header(models: dict[str, Any]) -> None:
    mode_label = _model_b_mode_label(st.session_state.get("model_b_mode", "middle_ground"))
    st.markdown(
        f"""
        <div class="hero">
            <h1>RACE Reading Comprehension and Quiz Generation System</h1>
            <div>Model A generates and verifies questions. Model B generates distractors, hints, and analytics-ready outputs.</div>
            <div class="pill-row">
                <span class="pill">CPU-only classical ML</span>
                <span class="pill">Model A generation + verification</span>
                <span class="pill">Model B distractors + hints</span>
                <span class="pill">Word2Vec + one-hot retrieval</span>
                <span class="pill">{mode_label}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    cols = st.columns(4)
    with cols[0]:
        st.metric("Question ranker", "Ready" if models.get("question_ranker") is not None else "Fallback")
    with cols[1]:
        st.metric("Distractor ranker", "Ready" if models.get("distractor_ranker") is not None else "Fallback")
    with cols[2]:
        st.metric("Hint scorer", "Ready" if models.get("hint_scorer") is not None else "Fallback")
    with cols[3]:
        st.metric("Word2Vec", "Ready" if models.get("w2v_model") is not None else "Fallback")


def render_article_input(models: dict[str, Any]) -> None:
    st.subheader("Screen 1 - Article Input")
    st.write("Paste a passage or load a random RACE sample, then submit to generate the quiz, distractors, and hints.")

    left, right = st.columns(2, gap="large")
    with left:
        if st.button("Load Random RACE Sample", use_container_width=True):
            try:
                sample = _pick_random_sample()
                st.session_state.article_input = sample["article"]
                st.session_state.question_input = sample["question"]
                st.session_state.correct_answer_input = sample["correct_answer"]
                st.session_state.last_sample_id = sample["sample_id"]
                st.success("Loaded a random RACE sample.")
            except Exception as exc:
                st.error(f"Could not load a sample: {exc}")
    with right:
        uploaded_file = st.file_uploader("Upload a passage file", type=["txt", "md", "csv"])
        if uploaded_file is not None:
            try:
                uploaded_text = uploaded_file.getvalue().decode("utf-8", errors="ignore")
                if uploaded_text.strip():
                    st.session_state.article_input = uploaded_text
                    st.success("Uploaded passage loaded into the article field.")
                else:
                    st.warning("The uploaded file could not be read as text.")
            except Exception:
                st.warning("The uploaded file could not be read as text.")

    with st.container(border=True):
        article = st.text_area("Reading passage", value=st.session_state.article_input, height=260, placeholder="Paste the article here...")
        question_seed = st.text_input("Optional existing question", value=st.session_state.question_input, placeholder="Leave blank to let Model A generate the question")
        correct_answer = st.text_input("Correct answer text", value=st.session_state.correct_answer_input, placeholder="Required for generation and distractors")

    st.session_state.article_input = article
    st.session_state.question_input = question_seed
    st.session_state.correct_answer_input = correct_answer

    if st.button("Submit", type="primary", use_container_width=True):
        if not article.strip():
            st.error("Please paste or upload a passage before submitting.")
            return
        if not correct_answer.strip():
            st.error("Please provide the correct answer text or load a sample.")
            return
        with st.spinner("Running Model A generation and Model B retrieval..."):
            quiz = _build_quiz(article, correct_answer, question_seed, models, st.session_state.get("model_b_mode", "middle_ground"))
        st.session_state.current_quiz = quiz
        st.session_state.quiz_version += 1
        st.session_state.hint_used_1 = False
        st.session_state.hint_used_2 = False
        st.session_state.hint_used_3 = False
        st.session_state.selected_option = None
        st.session_state.last_inference = quiz["timings"]
        st.success("Quiz generated. Open the Quiz View tab to check the answer and review hints.")


def render_quiz_view(models: dict[str, Any]) -> None:
    st.subheader("Screen 2 - Question & Answer Quiz View")
    quiz = st.session_state.current_quiz
    if not quiz:
        st.info("Submit a passage first to generate a question and answer options.")
        return

    st.markdown(f"<div class='panel-card'><strong>Generated question:</strong><br>{quiz['generated_question']}</div>", unsafe_allow_html=True)
    if quiz.get("original_question"):
        st.caption(f"Original sample question: {quiz['original_question']}")

    option_texts = quiz["options"]
    selection = st.radio(
        "Choose one answer",
        ["A", "B", "C", "D"],
        format_func=lambda label: f"{label}. {option_texts.get(label, '')}",
        key=f"quiz_selection_{st.session_state.quiz_version}",
    )

    if st.button("Check Answer", type="primary", use_container_width=True):
        try:
            start = time.perf_counter()
            scored = _score_options(st.session_state.article_input, quiz["generated_question"], option_texts, models)
            verification_time = time.perf_counter() - start
            predicted_label = str(scored.iloc[0]["option_label"])
            selected_correct = selection == quiz["correct_label"]
            verifier_correct = predicted_label == quiz["correct_label"]
            _store_session_result(quiz, selection, predicted_label, verifier_correct, selected_correct, verification_time)

            if selected_correct:
                st.success(f"Correct. Model A predicted {predicted_label} and your selection matches the key.")
            else:
                st.error(f"Incorrect. Model A predicted {predicted_label}; the correct label is {quiz['correct_label']}.")

            if quiz.get("source_sentence"):
                st.markdown(f"<div class='panel-card'>{_highlight_sentence(st.session_state.article_input, quiz['source_sentence'])}</div>", unsafe_allow_html=True)
            st.caption(f"Verification latency: {verification_time:.3f}s")
        except Exception as exc:
            st.error(f"Model verification failed: {exc}")


def render_hint_panel(models: dict[str, Any]) -> None:
    st.subheader("Screen 3 - Hint Panel")
    quiz = st.session_state.current_quiz
    if not quiz:
        st.info("Generate a quiz first to see hints.")
        return

    hint1, hint2, hint3 = quiz.get("hints", ["", "", ""])
    with st.expander("Hint 1 - General Clue", expanded=False):
        st.write(hint1)
        st.session_state.hint_used_1 = st.checkbox("I used Hint 1", value=st.session_state.hint_used_1, key=f"hint1_{st.session_state.quiz_version}")
    with st.expander("Hint 2 - More Specific", expanded=False):
        st.write(hint2)
        st.session_state.hint_used_2 = st.checkbox("I used Hint 2", value=st.session_state.hint_used_2, key=f"hint2_{st.session_state.quiz_version}")
    with st.expander("Hint 3 - Near Explicit", expanded=False):
        st.write(hint3)
        st.session_state.hint_used_3 = st.checkbox("I used Hint 3", value=st.session_state.hint_used_3, key=f"hint3_{st.session_state.quiz_version}")

    if st.session_state.hint_used_1 and st.session_state.hint_used_2 and st.session_state.hint_used_3:
        if st.button("Reveal Answer", use_container_width=True):
            st.success(f"The correct answer is: {quiz['correct_answer']}")
    else:
        st.caption("Open and use all three hints before revealing the answer.")


def _binary_metrics_from_log(entries: list[dict[str, Any]]) -> dict[str, Any]:
    if not entries:
        return {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0, "confusion_matrix": [[0, 0], [0, 0]]}
    y_true = [1 if item.get("selected_correct") else 0 for item in entries]
    y_pred = [1 if item.get("verifier_correct") else 0 for item in entries]
    tp = sum(1 for truth, pred in zip(y_true, y_pred) if truth == 1 and pred == 1)
    tn = sum(1 for truth, pred in zip(y_true, y_pred) if truth == 0 and pred == 0)
    fp = sum(1 for truth, pred in zip(y_true, y_pred) if truth == 0 and pred == 1)
    fn = sum(1 for truth, pred in zip(y_true, y_pred) if truth == 1 and pred == 0)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = (2 * precision * recall / max(precision + recall, 1e-12)) if (precision or recall) else 0.0
    return {
        "accuracy": float((tp + tn) / max(len(entries), 1)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
    }


def _confusion_figure(matrix: list[list[int]]):
    return pd.DataFrame(
        matrix,
        index=["True wrong", "True correct"],
        columns=["Pred wrong", "Pred correct"],
    )


def _latency_figure(entries: list[dict[str, Any]]):
    if not entries:
        return pd.DataFrame(columns=["Mean seconds"])
    df = pd.DataFrame(entries)
    metrics = {
        "Question generation": df["model_a_latency"].mean(),
        "Distractors + hints": df["model_b_latency"].mean(),
        "Verification": df["verification_latency"].mean(),
    }
    return pd.DataFrame({"Mean seconds": metrics}).T


def _metric_cards(metrics: dict[str, float], prefix: str = "") -> None:
    cols = st.columns(4)
    fields = [("Accuracy", metrics.get("accuracy", 0.0)), ("Precision", metrics.get("precision", 0.0)), ("Recall", metrics.get("recall", 0.0)), ("F1", metrics.get("f1", 0.0))]
    for column, (label, value) in zip(cols, fields):
        with column:
            st.metric(f"{prefix}{label}", f"{value:.2f}")


def _recursively_find_metric_block(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {}
    if any(key in payload for key in ["accuracy", "f1_macro", "macro_f1", "precision_macro", "recall_macro", "precision", "recall", "f1"]):
        return payload
    for value in payload.values():
        found = _recursively_find_metric_block(value)
        if found:
            return found
    return {}


def render_header(models: dict[str, Any]) -> None:
    st.markdown(
        """
        <div class="hero">
            <h1>RACE Reading Comprehension and Quiz Generation System</h1>
            <div>Model A generates and verifies questions. Model B generates distractors, hints, and analytics-ready outputs.</div>
            <div class="pill-row">
                <span class="pill">CPU-only classical ML</span>
                <span class="pill">Model A generation + verification</span>
                <span class="pill">Model B distractors + hints</span>
                <span class="pill">Word2Vec + one-hot retrieval</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    cols = st.columns(4)
    with cols[0]:
        st.metric("Question ranker", "Ready" if models.get("question_ranker") is not None else "Fallback")
    with cols[1]:
        st.metric("Distractor ranker", "Ready" if models.get("distractor_ranker") is not None else "Fallback")
    with cols[2]:
        st.metric("Hint scorer", "Ready" if models.get("hint_scorer") is not None else "Fallback")
    with cols[3]:
        st.metric("Word2Vec", "Ready" if models.get("w2v_model") is not None else "Fallback")


def render_analytics_dashboard(models: dict[str, Any]) -> None:
    st.subheader("Screen 4 - Developer / Analytics Dashboard")
    log_entries = st.session_state.session_log
    tabs = st.tabs(["Model A", "Model B", "Session Log", "Latency"])

    with tabs[0]:
        recent_entries = log_entries[-20:]
        model_a_metrics = _binary_metrics_from_log(recent_entries)
        _metric_cards(model_a_metrics, prefix="Model A ")
        confusion_view = _confusion_figure(model_a_metrics["confusion_matrix"])
        st.dataframe(confusion_view, use_container_width=True)
        st.bar_chart(confusion_view)
        summary_block = _recursively_find_metric_block(models.get("summary", {}))
        if summary_block:
            st.caption("Loaded summary metrics from results_summary.json")
            st.json(summary_block)
        elif models.get("model_a_eval"):
            st.caption("Loaded metrics from models/model_a/traditional/evaluation_results.json")
            st.json(_recursively_find_metric_block(models["model_a_eval"]))

    with tabs[1]:
        model_b_eval = models.get("model_b_eval", {})
        if model_b_eval:
            st.json(model_b_eval)
        else:
            st.info("Run src/model_b_train.py to create the Model B evaluation summary.")

    with tabs[2]:
        if log_entries:
            log_frame = pd.DataFrame(log_entries)
            st.dataframe(log_frame, use_container_width=True, hide_index=True)
            st.download_button("Export CSV", data=log_frame.to_csv(index=False).encode("utf-8"), file_name="session_log.csv", mime="text/csv", use_container_width=True)
        else:
            st.info("No answered quizzes have been logged yet.")

    with tabs[3]:
        latency_view = _latency_figure(log_entries[-20:])
        st.dataframe(latency_view, use_container_width=True)
        st.bar_chart(latency_view)
        if log_entries:
            df = pd.DataFrame(log_entries[-20:])
            st.dataframe(
                pd.DataFrame(
                    {
                        "Component": ["Question generation", "Distractors + hints", "Verification"],
                        "Mean seconds": [df["model_a_latency"].mean(), df["model_b_latency"].mean(), df["verification_latency"].mean()],
                    }
                ),
                use_container_width=True,
                hide_index=True,
            )


def main() -> None:
    apply_theme()
    _ensure_state_defaults()

    with st.spinner("Loading models and caches..."):
        models = load_models()

    st.sidebar.subheader("Model B Mode")
    st.session_state.model_b_mode = st.sidebar.radio(
        "Choose the Model B retrieval path",
        options=["middle_ground", "one_hot"],
        format_func=_model_b_mode_label,
        index=0 if st.session_state.get("model_b_mode", "middle_ground") == "middle_ground" else 1,
    )

    render_header(models)
    st.markdown("---")
    tabs = st.tabs(["Article Input", "Quiz View", "Hint Panel", "Analytics Dashboard"])
    with tabs[0]:
        render_article_input(models)
    with tabs[1]:
        render_quiz_view(models)
    with tabs[2]:
        render_hint_panel(models)
    with tabs[3]:
        render_analytics_dashboard(models)


if __name__ == "__main__":
    main()
