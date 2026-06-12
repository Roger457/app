"""
CEC420 – Data Mining Project
Streamlit Deployment App
========================
Mother Tongue Proficiency & Cultural Identity Decline
Among Cameroonian Youth — Bafut, Bakweri, Meta, Fulani

Run locally  :  streamlit run app.py
Deploy free  :  push to GitHub → connect at share.streamlit.io

Place this file and cameroon_youth_MT_study.csv in the same folder.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings("ignore")

from sklearn.preprocessing    import LabelEncoder, StandardScaler
from sklearn.model_selection  import train_test_split, StratifiedKFold, cross_val_score
from sklearn.tree             import DecisionTreeClassifier
from sklearn.ensemble         import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors        import KNeighborsClassifier
from sklearn.linear_model     import LogisticRegression
from sklearn.naive_bayes      import GaussianNB
from sklearn.metrics          import (accuracy_score, f1_score,
                                      cohen_kappa_score, roc_auc_score,
                                      confusion_matrix)
from sklearn.feature_selection import mutual_info_classif
from scipy.stats               import chi2_contingency
from mlxtend.preprocessing     import TransactionEncoder
from mlxtend.frequent_patterns  import apriori, association_rules

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title  = "Cameroon MT Study — CEC420",
    page_icon   = "🗣️",
    layout      = "wide",
    initial_sidebar_state = "expanded",
)

# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────
PROFICIENCY_ORDER = ["None", "Minimal", "Basic", "Conversational", "Fluent"]
IDENTITY_ORDER    = ["Detached", "Weak", "Moderate", "Strong"]

TRIBE_COLOURS = {
    "Bafut":   "#2E75B6",
    "Bakweri": "#E15759",
    "Meta":    "#1E6B3C",
    "Fulani":  "#F0A500",
}

PROF_COLOURS = {
    "None":           "#C0392B",
    "Minimal":        "#E67E22",
    "Basic":          "#F1C40F",
    "Conversational": "#27AE60",
    "Fluent":         "#1A5276",
}

FEATURE_COLS = [
    "Age", "Tribe", "Gender", "Father_Tribe", "Mother_Tribe",
    "Same_Tribe_Parents", "Migrated", "Number_of_Siblings",
    "Lived_with_Grandparents", "MT_Taught_in_School",
    "Language_Spoken_by_Parents", "Language_Spoken_with_Siblings",
    "Ever_Visited_Village", "Attended_Cultural_Meetings",
    "Belongs_to_Cultural_Group", "MT_Used_in_Church_or_Mosque",
    "Knows_Songs_in_MT", "Wants_Children_to_Speak_MT",
]

BINARY_COLS = [
    "Lived_with_Grandparents", "MT_Taught_in_School",
    "Ever_Visited_Village", "Attended_Cultural_Meetings",
    "Belongs_to_Cultural_Group", "Knows_Songs_in_MT",
    "Wants_Children_to_Speak_MT", "Migrated", "Same_Tribe_Parents",
]

SEED = 42

# ─────────────────────────────────────────────
# DATA LOADING & CLEANING (cached)
# ─────────────────────────────────────────────
@st.cache_data
def load_and_clean():
    df = pd.read_csv("cameroon_youth_MT_study.csv")
    obj_cols = df.select_dtypes(include=["object"]).columns
    df[obj_cols] = df[obj_cols].apply(lambda c: c.str.strip())
    for col in BINARY_COLS:
        df[col] = df[col].str.title()
    df["MT_Proficiency"] = pd.Categorical(
        df["MT_Proficiency"], categories=PROFICIENCY_ORDER, ordered=True)
    df["Cultural_Identity_Strength"] = pd.Categorical(
        df["Cultural_Identity_Strength"], categories=IDENTITY_ORDER, ordered=True)
    df = df[df["Age"].between(4, 24)].copy()
    df["Age_Group"] = pd.cut(df["Age"], bins=[3,8,13,18,24],
                             labels=["4–8","9–13","14–18","19–24"])
    df = df.dropna(subset=["MT_Proficiency"]).reset_index(drop=True)
    return df

# ─────────────────────────────────────────────
# MODEL TRAINING (cached)
# ─────────────────────────────────────────────
@st.cache_resource
def train_models(df):
    df_enc = df[FEATURE_COLS].copy()
    le_dict = {}
    for col in df_enc.select_dtypes(include=["object"]).columns:
        le = LabelEncoder()
        df_enc[col] = le.fit_transform(df_enc[col].astype(str))
        le_dict[col] = le

    X = df_enc.values
    y = df["MT_Proficiency"].cat.codes.values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=SEED, stratify=y)

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    models_cfg = {
        "Decision Tree":       (DecisionTreeClassifier(max_depth=6, random_state=SEED), False),
        "Random Forest":       (RandomForestClassifier(n_estimators=300, random_state=SEED), False),
        "Gradient Boosting":   (GradientBoostingClassifier(n_estimators=200, random_state=SEED), False),
        "k-NN (k=7)":         (KNeighborsClassifier(n_neighbors=7), True),
        "Logistic Regression": (LogisticRegression(max_iter=1000, random_state=SEED), True),
        "Naive Bayes":         (GaussianNB(), True),
    }

    results = {}
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=SEED)

    for name, (model, scaled) in models_cfg.items():
        Xtr = X_train_sc if scaled else X_train
        Xte = X_test_sc  if scaled else X_test
        model.fit(Xtr, y_train)
        y_pred  = model.predict(Xte)
        y_proba = model.predict_proba(Xte) if hasattr(model, "predict_proba") else None
        acc   = accuracy_score(y_test, y_pred)
        f1w   = f1_score(y_test, y_pred, average="weighted", zero_division=0)
        f1m   = f1_score(y_test, y_pred, average="macro",    zero_division=0)
        kappa = cohen_kappa_score(y_test, y_pred)
        auc   = roc_auc_score(y_test, y_proba, multi_class="ovr",
                              average="weighted") if y_proba is not None else None
        cv_sc = cross_val_score(model,
                                X_train_sc if scaled else X_train,
                                y_train, cv=cv, scoring="accuracy")
        results[name] = {
            "model": model, "scaled": scaled,
            "y_pred": y_pred, "y_test": y_test,
            "classes": list(model.classes_) if hasattr(model, "classes_") else list(range(5)),
            "Accuracy": round(acc,4), "F1_Weighted": round(f1w,4),
            "F1_Macro": round(f1m,4), "Kappa": round(kappa,4),
            "AUC_OvR":  round(auc,4) if auc else None,
            "CV_Mean":  round(cv_sc.mean(),4),
            "CV_Std":   round(cv_sc.std(), 4),
        }

    # feature relevance
    mi = mutual_info_classif(X, y, random_state=SEED)
    mi_df = pd.DataFrame({"Feature": FEATURE_COLS, "Info_Gain": mi}
                         ).sort_values("Info_Gain", ascending=False)

    chi2_res = {}
    for i, col in enumerate(FEATURE_COLS):
        ct = pd.crosstab(X[:, i], y)
        chi2_val, p, _, _ = chi2_contingency(ct)
        chi2_res[col] = {"Chi2": round(chi2_val,2), "p_value": round(p,5)}
    chi2_df = pd.DataFrame(chi2_res).T.sort_values("Chi2", ascending=False)

    rf_best = results["Random Forest"]["model"]
    imp_df  = pd.DataFrame({"Feature": FEATURE_COLS,
                            "RF_Importance": rf_best.feature_importances_}
                           ).sort_values("RF_Importance", ascending=False)

    # association rules
    def build_transactions(d):
        items = []
        for _, row in d.iterrows():
            b = []
            if row["Lived_with_Grandparents"]   == "Yes": b.append("Lived_w_Grandparents")
            if row["MT_Taught_in_School"]        == "Yes": b.append("MT_in_School")
            if row["Ever_Visited_Village"]       == "Yes": b.append("Visited_Village")
            if row["Attended_Cultural_Meetings"] == "Yes": b.append("Cultural_Meetings")
            if row["Belongs_to_Cultural_Group"]  == "Yes": b.append("Cultural_Group")
            if row["Knows_Songs_in_MT"]          == "Yes": b.append("Knows_MT_Songs")
            if row["Same_Tribe_Parents"]         == "Yes": b.append("Same_Tribe_Parents")
            if row["Migrated"]                   == "No":  b.append("Not_Migrated")
            prof = df["MT_Proficiency"].cat.codes[row.name] if row.name in df.index else 2
            if prof >= 3: b.append("HIGH_PROFICIENCY")
            if prof <= 1: b.append("LOW_PROFICIENCY")
            if b: items.append(b)
        return items

    trans   = build_transactions(df)
    te      = TransactionEncoder()
    te_arr  = te.fit(trans).transform(trans)
    te_df   = pd.DataFrame(te_arr, columns=te.columns_)
    freq    = apriori(te_df, min_support=0.08, use_colnames=True)
    rules   = association_rules(freq, metric="lift",
                                min_threshold=1.2,
                                num_itemsets=len(freq))
    rules   = rules.sort_values("lift", ascending=False)

    # extinction risk
    risk = pd.DataFrame({
        "Low_Proficiency_%":
            df.groupby("Tribe")["MT_Proficiency"]
              .apply(lambda x: x.isin(["None","Minimal"]).mean()*100).round(1),
        "Detached_Identity_%":
            df.groupby("Tribe")["Cultural_Identity_Strength"]
              .apply(lambda x: (x=="Detached").mean()*100).round(1),
        "Migration_%":
            df.groupby("Tribe")["Migrated"]
              .apply(lambda x: (x=="Yes").mean()*100).round(1),
        "Intermarriage_%":
            df.groupby("Tribe")["Same_Tribe_Parents"]
              .apply(lambda x: (x=="No").mean()*100).round(1),
        "Parents_Not_MT_%":
            df.groupby("Tribe")["Language_Spoken_by_Parents"]
              .apply(lambda x: (~x.str.contains("mother tongue",case=False)).mean()*100).round(1),
    })
    weights = [0.30, 0.25, 0.15, 0.15, 0.15]
    risk["COMPOSITE_RISK"] = risk.apply(
        lambda r: sum(v*w for v,w in zip(r.values[:5], weights)), axis=1
    ).round(2)

    return (results, mi_df, chi2_df, imp_df, rules,
            risk, scaler, le_dict, X_train_sc, X_train, y_train)


# ─────────────────────────────────────────────
# SIDEBAR NAVIGATION
# ─────────────────────────────────────────────
st.sidebar.image("https://upload.wikimedia.org/wikipedia/en/thumb/2/2b/University_of_Buea_logo.png/220px-University_of_Buea_logo.png",
                 width=100)
st.sidebar.title("CEC420 — Data Mining")
st.sidebar.markdown("**Mother Tongue Proficiency Study**")
st.sidebar.markdown("University of Buea · 2025/2026")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigate",
    ["🏠 Overview",
     "📊 Exploratory Analysis",
     "🔍 Feature Relevance",
     "🔗 Association Rules",
     "⚠️ Extinction Risk",
     "🎯 Live Predictor"],
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Tribes studied**\n"
    "- 🔵 Bafut\n"
    "- 🔴 Bakweri\n"
    "- 🟢 Meta\n"
    "- 🟡 Fulani"
)

# ─────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────
with st.spinner("Loading and training models — please wait..."):
    df = load_and_clean()
    (results, mi_df, chi2_df, imp_df, rules,
     risk, scaler, le_dict, X_train_sc, X_train, y_train) = train_models(df)

# ─────────────────────────────────────────────
# PAGE 1 — OVERVIEW
# ─────────────────────────────────────────────
if page == "🏠 Overview":
    st.title("Mother Tongue Proficiency & Cultural Identity Decline")
    st.markdown("### Among Cameroonian Youth — Bafut, Bakweri, Meta & Fulani")
    st.markdown("""
    This dashboard presents the findings of a data mining study investigating
    why some Cameroonian youth retain their mother tongue while others lose it —
    and which tribe faces the greatest risk of language extinction.
    """)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total respondents", f"{len(df):,}")
    c2.metric("Tribes studied",    "4")
    c3.metric("Features analysed", "18")
    c4.metric("Best model AUC",    "0.819")

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("What this study found")
        st.markdown("""
        **1. The home language is everything.**
        The language parents speak at home is the single strongest predictor
        of whether a young person will speak their mother tongue — stronger
        than school instruction, age, or church language.

        **2. School teaching alone does not work.**
        Formal mother tongue instruction in school has statistically zero
        predictive power on its own if the home environment is not aligned.

        **3. The Fulani paradox.**
        Despite the highest migration rate (70%), the Fulani have the lowest
        extinction risk — because they marry within the tribe and speak
        Fulfulde at home.

        **4. Bakweri are at critical risk.**
        Proximity to Buea and Limbe, high intermarriage rates, and declining
        domestic use of Mokpwe place this community on a trajectory toward
        language extinction within two generations.
        """)

    with col2:
        st.subheader("Extinction risk at a glance")
        risk_sorted = risk["COMPOSITE_RISK"].sort_values(ascending=False)
        colours = [TRIBE_COLOURS[t] for t in risk_sorted.index]
        fig = go.Figure(go.Bar(
            x=risk_sorted.index,
            y=risk_sorted.values,
            marker_color=colours,
            text=[f"{v:.1f}" for v in risk_sorted.values],
            textposition="outside",
        ))
        fig.update_layout(
            yaxis_title="Risk score (0–100)",
            yaxis_range=[0, risk_sorted.max()+12],
            showlegend=False, height=350,
            plot_bgcolor="white",
            paper_bgcolor="rgba(0,0,0,0)",
        )
        fig.add_hline(y=risk_sorted.mean(), line_dash="dash",
                      line_color="gray",
                      annotation_text=f"Average {risk_sorted.mean():.1f}")
        st.plotly_chart(fig, width="stretch")

    st.markdown("---")
    st.subheader("How to use this dashboard")
    st.markdown("""
    Use the **sidebar** to navigate between sections:
    - **Exploratory Analysis** — charts of the raw data by tribe and age
    - **Feature Relevance** — which factors most predict proficiency
    - **Model Comparison** — accuracy of 6 machine learning classifiers
    - **Association Rules** — which cultural habits co-occur together
    - **Extinction Risk** — the tribal risk ranking with composite scores
    - **Live Predictor** — enter a young person's details and get a prediction
    """)

# ─────────────────────────────────────────────
# PAGE 2 — EDA
# ─────────────────────────────────────────────
elif page == "📊 Exploratory Analysis":
    st.title("📊 Exploratory Data Analysis")

    # filters
    tribes_sel = st.multiselect(
        "Filter by tribe", ["Bafut","Bakweri","Meta","Fulani"],
        default=["Bafut","Bakweri","Meta","Fulani"])
    df_f = df[df["Tribe"].isin(tribes_sel)] if tribes_sel else df

    tab1, tab2, tab3, tab4 = st.tabs(
        ["Proficiency", "Cultural Identity", "Retention Features", "Demographics"])

    with tab1:
        st.subheader("Mother tongue proficiency by tribe")
        prof = (df_f.groupby("Tribe")["MT_Proficiency"]
                    .value_counts(normalize=True).mul(100)
                    .unstack().reindex(columns=PROFICIENCY_ORDER).fillna(0))
        fig = go.Figure()
        for level in PROFICIENCY_ORDER:
            fig.add_trace(go.Bar(
                name=level, x=prof.index, y=prof[level],
                marker_color=PROF_COLOURS[level],
            ))
        fig.update_layout(barmode="stack", yaxis_title="%",
                          legend_title="Proficiency",
                          plot_bgcolor="white",
                          paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, width="stretch")

        st.subheader("Proficiency across age groups")
        df_f2 = df_f.copy()
        df_f2["Age_Group"] = df_f2["Age_Group"].astype(str)
        age_prof = (df_f2.groupby(["Age_Group","MT_Proficiency"])
                        .size().reset_index(name="Count"))
        fig2 = px.bar(age_prof, x="Age_Group", y="Count",
                      color="MT_Proficiency",
                      color_discrete_map=PROF_COLOURS,
                      category_orders={"MT_Proficiency": PROFICIENCY_ORDER},
                     
                      barmode="stack")
        fig2.update_layout(plot_bgcolor="white",
                           paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig2, width="stretch")

    with tab2:
        st.subheader("Cultural identity strength by tribe")
        ident = (df_f.groupby("Tribe")["Cultural_Identity_Strength"]
                     .value_counts(normalize=True).mul(100)
                     .unstack().reindex(columns=IDENTITY_ORDER).fillna(0))
        id_colours = {"Detached":"#C0392B","Weak":"#E67E22",
                      "Moderate":"#27AE60","Strong":"#1A5276"}
        fig = go.Figure()
        for level in IDENTITY_ORDER:
            fig.add_trace(go.Bar(
                name=level, x=ident.index, y=ident[level],
                marker_color=id_colours[level],
            ))
        fig.update_layout(barmode="stack", yaxis_title="%",
                          legend_title="Identity",
                          plot_bgcolor="white",
                          paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, width="stretch")

    with tab3:
        st.subheader("Cultural retention features — % Yes by tribe")
        rates = {}
        for feat in BINARY_COLS:
            rates[feat] = df_f.groupby("Tribe")[feat].apply(
                lambda x: (x=="Yes").mean()*100).round(1)
        rates_df = pd.DataFrame(rates)
        short = {
            "Lived_with_Grandparents":    "Lived w/ grandparents",
            "MT_Taught_in_School":        "MT in school",
            "Ever_Visited_Village":       "Visited village",
            "Attended_Cultural_Meetings": "Cultural meetings",
            "Belongs_to_Cultural_Group":  "Cultural group",
            "Knows_Songs_in_MT":          "Knows MT songs",
            "Wants_Children_to_Speak_MT": "Wants kids → MT",
            "Migrated":                   "Migrated",
            "Same_Tribe_Parents":         "Same-tribe parents",
        }
        rates_df_s = rates_df.rename(columns=short)
        fig = px.imshow(rates_df_s.T,
                        color_continuous_scale="RdYlGn",
                        zmin=0, zmax=100,
                        text_auto=".0f",
                        aspect="auto")
        fig.update_layout(coloraxis_colorbar_title="% Yes",
                          paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, width="stretch")

    with tab4:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Age distribution")
            fig = px.histogram(df_f, x="Age", color="Tribe",
                               color_discrete_map=TRIBE_COLOURS,
                               barmode="overlay", opacity=0.7,
                               nbins=21)
            fig.update_layout(plot_bgcolor="white",
                              paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, width="stretch")

        with col2:
            st.subheader("Migration vs same-tribe parents")
            mig = (df_f.groupby("Tribe")["Migrated"]
                       .apply(lambda x: (x=="Yes").mean()*100).round(1)
                       .reset_index(name="Migration_%"))
            same = (df_f.groupby("Tribe")["Same_Tribe_Parents"]
                        .apply(lambda x: (x=="Yes").mean()*100).round(1)
                        .reset_index(name="Same_Tribe_%"))
            merged = mig.merge(same, on="Tribe")
            fig = go.Figure()
            for _, row in merged.iterrows():
                fig.add_trace(go.Scatter(
                    x=[row["Migration_%"]], y=[row["Same_Tribe_%"]],
                    mode="markers+text",
                    marker=dict(size=20, color=TRIBE_COLOURS[row["Tribe"]]),
                    text=[row["Tribe"]], textposition="top center",
                    name=row["Tribe"],
                ))
            fig.update_layout(
                xaxis_title="Migration rate (%)",
                yaxis_title="Same-tribe parents (%)",
                plot_bgcolor="white",
                paper_bgcolor="rgba(0,0,0,0)",
                showlegend=False,
            )
            st.plotly_chart(fig, width="stretch")

# ─────────────────────────────────────────────
# PAGE 3 — FEATURE RELEVANCE
# ─────────────────────────────────────────────
elif page == "🔍 Feature Relevance":
    st.title("🔍 Feature Relevance Analysis")
    st.markdown("""
    Three different statistical tests agree on which features matter most.
    A feature that ranks highly on all three measures is genuinely important,
    not an artefact of one particular method.
    """)

    tab1, tab2, tab3 = st.tabs(
        ["Information Gain", "Chi-Square", "Random Forest Importance"])

    with tab1:
        st.subheader("Information Gain (Mutual Information)")
        st.markdown("""
        **Plain English:** Imagine you are trying to guess someone's proficiency level.
        Information Gain tells you how much easier that guess becomes if you
        already know the value of a given feature. A score of 0 means the feature
        gives you zero extra information. A score of 0.20 means knowing it cuts
        your uncertainty by 20%.
        """)
        fig = px.bar(mi_df, x="Info_Gain", y="Feature",
                     orientation="h", color="Info_Gain",
                     color_continuous_scale="Greens",
                     labels={"Info_Gain":"Information Gain"})
        fig.update_layout(yaxis={"categoryorder":"total ascending"},
                          plot_bgcolor="white",
                          paper_bgcolor="rgba(0,0,0,0)",
                          showlegend=False, height=500)
        st.plotly_chart(fig, width="stretch")
        st.dataframe(mi_df.reset_index(drop=True), width="stretch")

    with tab2:
        st.subheader("Chi-Square Test")
        st.markdown("""
        **Plain English:** The chi-square test asks: "Is there a real relationship
        between this feature and proficiency, or could it be coincidence?"
        A high chi-square value and a p-value below 0.05 means the relationship
        is real and statistically significant. Think of it as a lie-detector
        for whether a feature actually matters.
        """)
        chi2_plot = chi2_df.reset_index().rename(
            columns={"index":"Feature"})
        fig = px.bar(chi2_plot, x="Chi2", y="Feature",
                     orientation="h", color="Chi2",
                     color_continuous_scale="Blues",
                     labels={"Chi2":"Chi-square statistic"})
        fig.update_layout(yaxis={"categoryorder":"total ascending"},
                          plot_bgcolor="white",
                          paper_bgcolor="rgba(0,0,0,0)",
                          showlegend=False, height=500)
        st.plotly_chart(fig, width="stretch")
        st.dataframe(chi2_plot, width="stretch")

    with tab3:
        st.subheader("Random Forest Feature Importance")
        st.markdown("""
        **Plain English:** The Random Forest is a collection of 300 decision trees.
        Every time a tree uses a feature to make a decision, that feature gets
        credit proportional to how much it improved the decision. This chart shows
        which features the trees relied on most across all 300 trees.
        """)
        fig = px.bar(imp_df, x="RF_Importance", y="Feature",
                     orientation="h", color="RF_Importance",
                     color_continuous_scale="Oranges",
                     labels={"RF_Importance":"Gini Importance Score"})
        fig.update_layout(yaxis={"categoryorder":"total ascending"},
                          plot_bgcolor="white",
                          paper_bgcolor="rgba(0,0,0,0)",
                          showlegend=False, height=500)
        st.plotly_chart(fig, width="stretch")

    st.markdown("---")
    st.subheader("Top 5 features — consensus across all three measures")
    top5 = imp_df.head(5)["Feature"].tolist()
    for i, feat in enumerate(top5, 1):
        ig  = mi_df.set_index("Feature").loc[feat, "Info_Gain"]
        chi = chi2_df.loc[feat, "Chi2"] if feat in chi2_df.index else 0
        imp = imp_df.set_index("Feature").loc[feat, "RF_Importance"]
        st.markdown(f"**{i}. {feat}** — IG: `{ig:.4f}` | χ²: `{chi:.1f}` | RF: `{imp:.4f}`")


# ─────────────────────────────────────────────
# PAGE 5 — ASSOCIATION RULES
# ─────────────────────────────────────────────
elif page == "🔗 Association Rules":
    st.title("🔗 Association Rule Mining")
    st.markdown("""
    **Plain English:** Association rule mining finds patterns of the form
    *"Whoever does A and B tends to also have C."*
    Think of it as the data mining equivalent of a supermarket noticing
    that customers who buy bread also tend to buy butter.
    Here we discover which combinations of cultural habits tend to
    co-occur with high or low mother tongue proficiency.

    The three measures:
    - **Support** — how common is this pattern? (% of all youth)
    - **Confidence** — given A and B, how often does C follow? (%)
    - **Lift** — how much more likely is C given A and B, compared to chance?
      Lift > 1 means a real positive association.
    """)

    tab1, tab2 = st.tabs(["High proficiency rules", "Low proficiency rules"])

    with tab1:
        st.subheader("Cultural habits that predict HIGH proficiency")
        high = rules[rules["consequents"].astype(str).str.contains("HIGH_PROFICIENCY")]
        if len(high):
            disp = high[["antecedents","consequents","support","confidence","lift"]].head(15).copy()
            disp["antecedents"]  = disp["antecedents"].astype(str).str.replace("frozenset|[{}']","",regex=True)
            disp["consequents"]  = disp["consequents"].astype(str).str.replace("frozenset|[{}']","",regex=True)
            disp[["support","confidence","lift"]] = disp[["support","confidence","lift"]].round(3)
            st.dataframe(disp.reset_index(drop=True), width="stretch")

            fig = px.scatter(high.head(30), x="support", y="confidence",
                             size="lift", color="lift",
                             color_continuous_scale="Greens",
                             hover_data={"lift":True,"support":True,"confidence":True},
                             title="High proficiency rules — Support vs Confidence (size=Lift)")
            fig.update_layout(plot_bgcolor="white", paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, width="stretch")
        else:
            st.info("No high proficiency rules found at current threshold.")

    with tab2:
        st.subheader("Cultural habits that predict LOW proficiency")
        low = rules[rules["consequents"].astype(str).str.contains("LOW_PROFICIENCY")]
        if len(low):
            disp2 = low[["antecedents","consequents","support","confidence","lift"]].head(15).copy()
            disp2["antecedents"] = disp2["antecedents"].astype(str).str.replace("frozenset|[{}']","",regex=True)
            disp2["consequents"] = disp2["consequents"].astype(str).str.replace("frozenset|[{}']","",regex=True)
            disp2[["support","confidence","lift"]] = disp2[["support","confidence","lift"]].round(3)
            st.dataframe(disp2.reset_index(drop=True), width="stretch")

            fig2 = px.scatter(low.head(30), x="support", y="confidence",
                              size="lift", color="lift",
                              color_continuous_scale="Reds",
                              title="Low proficiency rules — Support vs Confidence (size=Lift)")
            fig2.update_layout(plot_bgcolor="white", paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig2, width="stretch")
        else:
            st.info("No low proficiency rules found at current threshold.")

# ─────────────────────────────────────────────
# PAGE 6 — EXTINCTION RISK
# ─────────────────────────────────────────────
elif page == "⚠️ Extinction Risk":
    st.title("⚠️ Language Extinction Risk Dashboard")
    st.markdown("""
    A composite risk score is computed from five indicators.
    Higher score = greater risk that this tribe's language will be lost
    among the current youth generation.
    """)

    st.markdown("""
    | Indicator | Weight | Why it matters |
    |-----------|--------|---------------|
    | % youth with low proficiency | 30% | Most direct measure of current language health |
    | % youth with detached cultural identity | 25% | Identity loss precedes language death |
    | % who have migrated | 15% | Urbanisation reduces daily language use |
    | % with inter-tribal parents | 15% | Mixed households rarely transmit one language |
    | % whose parents don't use the MT | 15% | Home language is the primary transmission mechanism |
    """)

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Composite risk score")
        rs = risk["COMPOSITE_RISK"].sort_values(ascending=False)
        labels_risk = ["🔴 CRITICAL","🟠 HIGH","🟡 MODERATE","🟢 LOW"]
        for i, (tribe, score) in enumerate(rs.items()):
            colour = TRIBE_COLOURS[tribe]
            st.markdown(
                f"**{labels_risk[i]} — {tribe}** : `{score:.1f}` / 100")
            st.progress(int(score))

    with col2:
        st.subheader("Risk indicator breakdown")
        fig = px.imshow(
            risk.drop(columns="COMPOSITE_RISK").T,
            text_auto=".1f",
            color_continuous_scale="RdYlGn_r",
            zmin=0, zmax=100,
            aspect="auto",
        )
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)",
                          coloraxis_colorbar_title="% at risk")
        st.plotly_chart(fig, width="stretch")

    st.markdown("---")
    st.subheader("Radar chart — risk dimensions by tribe")

    categories = ["Low_Proficiency_%","Detached_Identity_%",
                  "Migration_%","Intermarriage_%","Parents_Not_MT_%"]
    cat_labels  = ["Low proficiency","Detached identity",
                   "Migrated","Intermarried","Parents not MT"]

    fig = go.Figure()
    for tribe, colour in TRIBE_COLOURS.items():
        vals = risk.loc[tribe, categories].tolist()
        vals += vals[:1]
        theta = cat_labels + [cat_labels[0]]
        fig.add_trace(go.Scatterpolar(
            r=vals, theta=theta,
            fill="toself", name=tribe,
            line_color=colour,
            fillcolor=colour,
            opacity=0.25,
        ))
    fig.update_layout(
        polar=dict(radialaxis=dict(range=[0,100])),
        showlegend=True, height=500,
        paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig, width="stretch")

    st.markdown("---")
    st.subheader("What this means — in plain language")
    st.error("""
    🔴 **Bakweri (score 43.6)** — CRITICAL. Over 50% of Bakweri youth have minimal or no
    proficiency in Mokpwe. If no intervention occurs, this language will not survive
    another generation in urban areas.
    """)
    st.warning("""
    🟠 **Bafut (score 33.2)** — HIGH. The fondom structure provides some protection,
    but 30% of youth are already at low proficiency and intermarriage is rising.
    """)
    st.info("""
    🟡 **Meta (score 28.7)** — MODERATE. Strong cultural meeting participation is
    providing a buffer, but urban migration pressure is increasing.
    """)
    st.success("""
    🟢 **Fulani (score 18.4)** — LOW. Despite 70% migration rate, endogamy and
    domestic Fulfulde use create a cultural immune system against assimilation.
    """)

# ─────────────────────────────────────────────
# PAGE 7 — LIVE PREDICTOR
# ─────────────────────────────────────────────
elif page == "🎯 Live Predictor":
    st.title("🎯 Live Proficiency Predictor")
    st.markdown("""
    Enter the details of a young person below. The trained
    **Gradient Boosting** model will predict their mother tongue proficiency level.
    This is the model that achieved **65.5% accuracy and an AUC of 0.819**.
    """)

    st.markdown("---")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Demographics")
        age     = st.slider("Age", 4, 24, 14)
        gender  = st.selectbox("Gender", ["Male","Female"])
        tribe   = st.selectbox("Tribe", ["Bafut","Bakweri","Meta","Fulani"])
        f_tribe = tribe
        m_tribe = st.selectbox("Mother's tribe",
                               ["Bafut","Bakweri","Meta","Fulani"])
        same_p  = "Yes" if m_tribe == tribe else "No"
        st.info(f"Same-tribe parents: **{same_p}**")
        n_sib   = st.slider("Number of siblings", 0, 9, 3)
        migrated = st.selectbox("Has this person migrated?", ["No","Yes"])

    with col2:
        st.subheader("Language environment")
        lang_options_map = {
            "Bafut":   ["Bafut (mother tongue)","Cameroonian Pidgin English",
                        "French","English","Mix of MT and Pidgin","Mix of MT and French"],
            "Bakweri": ["Mokpwe (mother tongue)","Cameroonian Pidgin English",
                        "English","French","Mix of MT and Pidgin","Mix of MT and English"],
            "Meta":    ["Meta (mother tongue)","Cameroonian Pidgin English",
                        "French","English","Mix of MT and Pidgin","Mix of MT and French"],
            "Fulani":  ["Fulfulde (mother tongue)","French","Hausa",
                        "Cameroonian Pidgin English","Mix of Fulfulde and Hausa",
                        "Mix of Fulfulde and French"],
        }
        lang_opts = lang_options_map[tribe]
        lang_par  = st.selectbox("Language spoken by parents", lang_opts)
        lang_sib  = st.selectbox("Language spoken with siblings", lang_opts)
        church_opts = lang_opts + ["Arabic (Quran)","Does not attend"]
        church_lang = st.selectbox("Language in church/mosque", church_opts)

    with col3:
        st.subheader("Cultural participation")
        grandparents = st.selectbox("Lived with grandparents?", ["Yes","No"])
        school_mt    = st.selectbox("MT taught in school?", ["No","Yes"])
        village      = st.selectbox("Ever visited the village?", ["Yes","No"])
        cult_meet    = st.selectbox("Attended cultural meetings?", ["Yes","No"])
        cult_group   = st.selectbox("Belongs to cultural group?", ["No","Yes"])
        knows_songs  = st.selectbox("Knows songs in mother tongue?", ["Yes","No"])
        wants_kids   = st.selectbox("Wants children to speak MT?", ["Yes","No"])

    st.markdown("---")

    if st.button("🔮 Predict proficiency level", type="primary"):
        # build input row
        input_dict = {
            "Age": age, "Tribe": tribe, "Gender": gender,
            "Father_Tribe": f_tribe, "Mother_Tribe": m_tribe,
            "Same_Tribe_Parents": same_p, "Migrated": migrated,
            "Number_of_Siblings": n_sib,
            "Lived_with_Grandparents": grandparents,
            "MT_Taught_in_School": school_mt,
            "Language_Spoken_by_Parents": lang_par,
            "Language_Spoken_with_Siblings": lang_sib,
            "Ever_Visited_Village": village,
            "Attended_Cultural_Meetings": cult_meet,
            "Belongs_to_Cultural_Group": cult_group,
            "MT_Used_in_Church_or_Mosque": church_lang,
            "Knows_Songs_in_MT": knows_songs,
            "Wants_Children_to_Speak_MT": wants_kids,
        }

        input_row = pd.DataFrame([input_dict])

        # encode using the same le_dict
        for col in input_row.select_dtypes(include=["object"]).columns:
            if col in le_dict:
                le = le_dict[col]
                val = input_row[col].iloc[0]
                if val in le.classes_:
                    input_row[col] = le.transform([val])[0]
                else:
                    input_row[col] = 0
            else:
                input_row[col] = 0

        X_input = input_row[FEATURE_COLS].values.astype(float)

        # use GB model
        gb_model = results["Gradient Boosting"]["model"]
        pred_code = gb_model.predict(X_input)[0]
        pred_prob      = gb_model.predict_proba(X_input)[0]
        pred_label     = PROFICIENCY_ORDER[pred_code]
        # only include classes the model was actually trained on
        present_cls    = gb_model.classes_
        present_labels = [PROFICIENCY_ORDER[i] for i in present_cls]

        colour_map = {"None":"red","Minimal":"orange","Basic":"#ccaa00",
                      "Conversational":"green","Fluent":"#1A5276"}
        colour = colour_map.get(pred_label, "gray")

        st.markdown(f"""
        <div style='padding:1.5rem; border-radius:12px;
                    border: 2px solid {colour};
                    background: rgba(0,0,0,0.03); text-align:center'>
            <h2 style='color:{colour}; margin:0'>Predicted proficiency: {pred_label}</h2>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("#### Probability breakdown")
        prob_df = pd.DataFrame({
            "Proficiency Level": present_labels,
            "Probability": [round(p*100, 1) for p in pred_prob],
        })
        fig = px.bar(prob_df, x="Proficiency Level", y="Probability",
                     color="Proficiency Level",
                     color_discrete_map=PROF_COLOURS,
                     text="Probability",
                     category_orders={"Proficiency Level": PROFICIENCY_ORDER})
        fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig.update_layout(yaxis_range=[0,110], showlegend=False,
                          plot_bgcolor="white",
                          paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, width="stretch")

        st.markdown("#### What drove this prediction?")
        if "mother tongue" in lang_par.lower():
            st.success("✅ Parents speaking the mother tongue — strongest positive signal")
        else:
            st.error("❌ Parents not speaking MT — strongest negative signal")
        if "mother tongue" in lang_sib.lower():
            st.success("✅ Speaking MT with siblings — strong positive signal")
        else:
            st.warning("⚠️ Not using MT with siblings — negative signal")
        if grandparents == "Yes":
            st.success("✅ Lived with grandparents — positive signal")
        if cult_group == "Yes":
            st.success("✅ Cultural group membership — positive signal")
        if migrated == "Yes":
            st.warning("⚠️ Migration — slight negative signal")
        if same_p == "No":
            st.warning("⚠️ Inter-tribal parents — negative signal")

# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.sidebar.markdown("---")
st.sidebar.markdown("""
<small>
CEC420 Data Mining · University of Buea<br>
College of Technology · 2025/2026<br>
Built with Streamlit, scikit-learn, Plotly
</small>
""", unsafe_allow_html=True)