import streamlit as st
import pandas as pd
import joblib

# -----------------------
# Load trained pipeline
# -----------------------
@st.cache_resource
def load_model():
    pipeline = joblib.load("models/sentiment_pipeline.pkl")
    return pipeline

pipeline = load_model()

# -----------------------
# Helper function
# -----------------------
def predict_single_review(review_text: str):
    """Predict sentiment for a single review string."""
    if not review_text or review_text.strip() == "":
        return None, None

    preds = pipeline.predict([review_text])
    label = preds[0]

    proba = None
    try:
        proba = pipeline.predict_proba([review_text])[0]
    except Exception:
        proba = None

    return label, proba


def predict_dataframe(df: pd.DataFrame, text_column: str):
    """Predict sentiment for all rows in a DataFrame column."""
    texts = df[text_column].astype(str).tolist()
    preds = pipeline.predict(texts)

    try:
        probas = pipeline.predict_proba(texts)
    except Exception:
        probas = None

    df_out = df.copy()
    df_out["predicted_sentiment"] = preds

    if probas is not None:
        # Assume 3 classes max; adjust as needed
        classes = list(pipeline.classes_)
        for i, cls in enumerate(classes):
            df_out[f"proba_{cls}"] = probas[:, i]

    return df_out


# -----------------------
# Streamlit UI
# -----------------------
st.set_page_config(
    page_title="Restaurant Review Sentiment",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 NLP Sentiment Analysis – Restaurant Reviews")
# Add header image here
st.image(
    "images/header.jpg",           
    caption="Restaurant Reviews - Sentiment Analysis",
    use_container_width=True
)
st.write(
    """
    This app uses a trained NLP model to predict whether a restaurant review is **positive, negative, or neutral**.
    You can:
    - Type a single review and get the sentiment.
    - Upload a CSV file with reviews and get batch predictions.
    """
)

st.sidebar.header("Options")
mode = st.sidebar.radio(
    "Choose mode:",
    ["Single Review", "Batch (CSV upload)"]
)

# -----------------------
# Single review mode
# -----------------------
if mode == "Single Review":
    st.subheader("🔹 Single Review Prediction")

    default_text = "The food was amazing and the staff were very friendly."
    review_text = st.text_area(
        "Enter a restaurant review:",
        value=default_text,
        height=150
    )

    if st.button("Predict Sentiment"):
        label, proba = predict_single_review(review_text)

        if label is None:
            st.warning("Please enter some text.")
        else:
            st.markdown(f"### ✅ Predicted Sentiment: **{label}**")

            if label.lower() == "positive":
                st.image("images/positive.png")
            elif label.lower() == "negative":
                st.image("images/negative.png")
            else:
                st.image("images/neutral.png")

            if proba is not None:
                st.write("**Prediction probabilities:**")
                classes = list(pipeline.classes_)
                proba_dict = {cls: float(p) for cls, p in zip(classes, proba)}
                st.json(proba_dict)

# -----------------------
# Batch mode
# -----------------------
else:
    st.subheader("📂 Batch Prediction from CSV")

    st.write(
        """
        1. Upload a CSV file (e.g. `restaurant.csv`).  
        2. Make sure it has a **text column** with reviews (e.g. `Review`).  
        3. Select the column name and click **Predict**.
        """
    )

    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("📄 Preview of uploaded file:")
        st.dataframe(df.head())

        # Guess a text column (try common names)
        default_col = None
        for col in df.columns:
            if col.lower() in ["review", "text", "comment"]:
                default_col = col
                break

        text_column = st.selectbox(
            "Select the column containing review text:",
            options=df.columns.tolist(),
            index=df.columns.tolist().index(default_col) if default_col in df.columns else 0
        )

        if st.button("Predict Sentiment for All Rows"):
            with st.spinner("Predicting..."):
                result_df = predict_dataframe(df, text_column)

            st.success("Prediction complete.")
            st.write("📊 Sample of results:")
            st.dataframe(result_df.head())

            # Option to download
            csv_out = result_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="⬇️ Download results as CSV",
                data=csv_out,
                file_name="sentiment_predictions.csv",
                mime="text/csv"
            )
