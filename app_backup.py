
import re
from pathlib import Path

import pandas as pd
import pymupdf
import streamlit as st
import altair as alt

from market_analysis import (
    load_job_postings,
    calculate_skill_demand,
    compare_job_roles,
)

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity



# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="ResumeIQ | Resume Analyzer",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)


# --------------------------------------------------
# CUSTOM CSS: LIGHT PROFESSIONAL DASHBOARD
# --------------------------------------------------

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background-color: #F4F8FC;
        color: #17233F;
    }

    [data-testid="stHeader"] {
        background: rgba(245, 247, 252, 0.95);
    }

    [data-testid="stSidebar"] {
        background: #FFFFFF;
        border-right: 1px solid #E6E9F2;
    }

    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
        color: #53617A;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1450px;
    }

    h1, h2, h3 {
        color: #17233F !important;
        letter-spacing: -0.5px;
    }

    p, li, label {
        color: #526079;
    }

    .brand {
        font-size: 25px;
        font-weight: 800;
        color: #6546D7;
        letter-spacing: -1px;
    }

    .eyebrow {
        color: #2457A7;
        font-size: 12px;
        font-weight: 700;
        letter-spacing: 1.5px;
        text-transform: uppercase;
    }

    .subtitle {
        color: #71809A;
        font-size: 14px;
        margin-top: -8px;
        margin-bottom: 24px;
    }

    .section-label {
        color: #7B879B;
        font-size: 12px;
        font-weight: 700;
        letter-spacing: 1px;
        text-transform: uppercase;
    }

    .info-card {
        background: #FFFFFF;
        border: 1px solid #E6E9F2;
        border-radius: 16px;
        padding: 20px;
        margin-bottom: 12px;
    }

    .metric-label {
        color: #75819A;
        font-size: 13px;
        font-weight: 600;
    }

    .metric-value {
        color: #17233F;
        font-size: 28px;
        font-weight: 800;
        margin-top: 8px;
    }

    .metric-note {
        color: #75819A;
        font-size: 12px;
        margin-top: 5px;
    }

    .skill-tag {
        display: inline-block;
        background: #F0ECFF;
        color: #5C3BC6;
        border: 1px solid #E1D8FF;
        padding: 6px 11px;
        border-radius: 8px;
        margin: 4px 5px 4px 0;
        font-size: 12px;
        font-weight: 600;
    }

    .matched-tag {
        display: inline-block;
        background: #E8F8F0;
        color: #16784A;
        padding: 6px 10px;
        border-radius: 7px;
        margin: 4px;
        font-size: 12px;
        font-weight: 600;
    }

    .missing-tag {
        display: inline-block;
        background: #FFF3E8;
        color: #A95C15;
        padding: 6px 10px;
        border-radius: 7px;
        margin: 4px;
        font-size: 12px;
        font-weight: 600;
    }

    div.stButton > button,
    div.stFormSubmitButton > button {
        background:#2457A7 ;
        color: white;
        border: none;
        border-radius: 9px;
        padding: 10px 18px;
        font-weight: 600;
        min-height: 42px;
    }

    div.stButton > button:hover,
    div.stFormSubmitButton > button:hover {
        background: #5234B7;
        color: white;
        border: none;
    }

    [data-testid="stMetric"] {
        background: #FFFFFF;
        border: 1px solid #E6E9F2;
        border-radius: 15px;
        padding: 18px 20px;
        box-shadow: 0 3px 12px rgba(28, 42, 76, 0.03);
    }

    [data-testid="stMetricLabel"] {
        color: #71809A;
    }

    [data-testid="stMetricValue"] {
        color: #17233F;
    }

    [data-baseweb="tab-list"] {
        gap: 8px;
        border-bottom: 1px solid #E4E8F1;
    }

    [data-baseweb="tab"] {
        background: transparent;
        border-radius: 8px 8px 0 0;
        color: #71809A;
        padding: 12px 17px;
    }

    [aria-selected="true"] {
        color: #2457A7 !important;
    }

    [data-testid="stFileUploader"] {
        background: #FFFFFF;
        border: 1px dashed #C8D8EB;
        border-radius: 12px;
        padding: 12px;
    }

    hr {
        border-color: #E6E9F2;
    }
    
/* Improve sidebar heading visibility */
[data-testid="stSidebar"] h4,
[data-testid="stSidebar"] h3 {
    color: #17233F !important;
    font-weight: 700;
}

/* Light-theme text inputs */
[data-testid="stTextArea"] textarea,
[data-testid="stTextInput"] input {
    background-color: #FFFFFF !important;
    color: #17233F !important;
    border: 1px solid #DCE2EF !important;
    border-radius: 10px !important;
}

[data-testid="stTextArea"] textarea::placeholder {
    color: #8490A5 !important;
}

/* Light-theme expander */
[data-testid="stExpander"] {
    background-color: #FFFFFF !important;
    border: 1px solid #E3E7F1 !important;
    border-radius: 12px !important;
}

[data-testid="stExpander"] summary {
    color: #17233F !important;
}

/* Compact job-description input */
[data-testid="stTextArea"] textarea {
    min-height: 110px;
}

/* Improve button visibility */
div.stFormSubmitButton > button {
    background-color: #6847D9 !important;
    color: #FFFFFF !important;
    font-weight: 700 !important;
}

/* Make sidebar captions readable */
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] {
    color: #68758D !important;
}

    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------
# HELPER FUNCTIONS
# --------------------------------------------------

SKILLS_FILE = Path(__file__).parent / "data" / "skills.csv"


@st.cache_data
def load_skills():
    if not SKILLS_FILE.exists():
        raise FileNotFoundError(
            "data/skills.csv was not found. "
            "Please check your project folder."
        )

    skills_df = pd.read_csv(SKILLS_FILE)

    if "skill" not in skills_df.columns:
        raise ValueError(
            "Your skills.csv file must contain a column named 'skill'."
        )

    return [
        str(skill).strip()
        for skill in skills_df["skill"].dropna()
        if str(skill).strip()
    ]


def detect_skills(text):
    all_skills = load_skills()
    found_skills = []

    for skill in all_skills:
        pattern = r"(?<!\w)" + re.escape(skill) + r"(?!\w)"

        if re.search(pattern, text, flags=re.IGNORECASE):
            found_skills.append(skill)

    return sorted(set(found_skills), key=str.lower)


def score_resume(text, skills):
    text_lower = text.lower()
    details = {}
    suggestions = []

    sections = {
        "Education": ["education", "academic background"],
        "Skills": ["skills", "technical skills"],
        "Projects": ["projects", "personal projects"],
        "Experience": ["experience", "internship", "work history"],
        "Contact Information": ["email", "phone", "linkedin", "github"],
    }

    section_score = 0

    for section, keywords in sections.items():
        if any(keyword in text_lower for keyword in keywords):
            section_score += 6
        else:
            suggestions.append(
                f"Consider adding a clear {section} section."
            )

    details["Resume Sections"] = section_score

    skill_score = min(len(skills) / 8, 1) * 25
    details["Skills"] = round(skill_score)

    if len(skills) < 5:
        suggestions.append(
            "Include more relevant skills that you genuinely possess."
        )

    action_verbs = [
        "developed", "built", "created", "designed",
        "implemented", "improved", "managed", "achieved",
        "analyzed", "automated", "optimized", "led",
    ]

    action_count = sum(
        len(re.findall(r"\b" + verb + r"\b", text_lower))
        for verb in action_verbs
    )

    result_count = len(re.findall(r"\b\d+%|\b\d+\+", text_lower))

    writing_score = min(action_count / 4, 1) * 10
    writing_score += min(result_count / 2, 1) * 5

    details["Action Verbs and Results"] = round(writing_score)

    if action_count < 2:
        suggestions.append(
            "Use action verbs such as Developed, Built, or Implemented."
        )

    if result_count == 0:
        suggestions.append(
            "Include genuine measurable results where possible."
        )

    word_count = len(text.split())

    if 300 <= word_count <= 1200:
        length_score = 10
    elif 150 <= word_count < 300 or 1200 < word_count <= 1800:
        length_score = 6
    else:
        length_score = 3

    details["Length"] = length_score

    if word_count < 300:
        suggestions.append(
            "Check whether all relevant resume content was extracted."
        )
    elif word_count > 1800:
        suggestions.append(
            "Consider removing repetitive or less relevant information."
        )

    contact_score = 0

    if re.search(r"\b[\w.+-]+@[\w.-]+\.\w+\b", text):
        contact_score += 5
    else:
        suggestions.append("Add a professional email address.")

    if re.search(r"linkedin\.com|github\.com", text_lower):
        contact_score += 5
    else:
        suggestions.append(
            "Consider adding a LinkedIn or GitHub link if relevant."
        )

    details["Contact and Links"] = contact_score

    readability_score = 5 if "\n" in text else 0
    readability_score += 5 if text.split() else 0

    details["Basic Readability"] = readability_score

    total_score = min(round(sum(details.values())), 100)

    return total_score, details, suggestions


def calculate_tfidf_similarity(resume_text, job_description):
    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
    )

    vectors = vectorizer.fit_transform(
        [resume_text, job_description]
    )

    similarity = cosine_similarity(
        vectors[0:1], vectors[1:2]
    )[0][0]

    return round(float(similarity) * 100)


def render_skill_tags(skills, css_class="skill-tag"):
    if not skills:
        st.write("No skills detected.")
        return

    tags = "".join(
        f'<span class="{css_class}">{skill}</span>'
        for skill in skills
    )

    st.markdown(tags, unsafe_allow_html=True)


def group_skills(skills):
    categories = {
        "Programming Languages": [
            "Python", "C++", "Java", "JavaScript",
        ],
        "Web Development": [
            "HTML", "CSS", "React", "Node.js",
            "Express", "Flask", "Django",
        ],
        "Databases": [
            "SQL", "MySQL", "PostgreSQL", "MongoDB",
        ],
        "AI and Data": [
            "Artificial Intelligence", "Machine Learning",
            "Natural Language Processing", "Pandas",
            "NumPy", "Scikit-learn",
        ],
        "Tools and Core Skills": [
            "Git", "GitHub", "Streamlit", "Data Structures",
            "Algorithms", "Communication", "Problem Solving",
        ],
    }

    grouped = {}

    for category, category_skills in categories.items():
        found = [
            skill for skill in skills
            if skill.lower() in {
                item.lower() for item in category_skills
            }
        ]

        if found:
            grouped[category] = found

    other_skills = [
        skill for skill in skills
        if not any(
            skill.lower() == item.lower()
            for items in categories.values()
            for item in items
        )
    ]

    if other_skills:
        grouped["Other Skills"] = other_skills

    return grouped


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

with st.sidebar:
    st.markdown(
        '<div class="brand">✦ ResumeIQ</div>',
        unsafe_allow_html=True,
    )

    st.caption("RESUME INTELLIGENCE PLATFORM")

    st.divider()

    st.markdown("### Your Workspace")
    st.write("Upload your resume to begin your analysis.")

    uploaded_file = st.file_uploader(
        "Upload Resume",
        type=["pdf"],
        help="Upload a text-based PDF resume.",
    )

    st.divider()

    st.markdown("#### Analysis Modules")
    st.markdown("📊  Resume Overview")
    st.markdown("🧩  Skill Intelligence")
    st.markdown("🎯  Job Matching")
    st.markdown("💡  Improvement Plan")

    st.divider()

    st.caption("ResumeIQ · Student Project")
    st.caption("Rule-based analysis and text similarity")


# --------------------------------------------------
# MAIN HEADER
# --------------------------------------------------

st.markdown(
    '<div class="eyebrow">CAREER INTELLIGENCE</div>',
    unsafe_allow_html=True,
)

st.title("Resume Analyzer")
st.markdown(
    '<div class="subtitle">'
    'Understand your resume, explore your skills, and compare '
    'your profile with job requirements.'
    '</div>',
    unsafe_allow_html=True,
)


# --------------------------------------------------
# EMPTY STATE
# --------------------------------------------------

if uploaded_file is None:
    st.markdown(
        """
        <div class="info-card">
            <h2>Welcome to ResumeIQ 👋</h2>
            <p>
                Start by uploading your resume PDF using the sidebar.
                Your dashboard will show resume quality indicators,
                detected skills, and job-description matching results.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            """
            <div class="info-card">
                <h3>📄 Resume Analysis</h3>
                <p>Extract resume text and review a preliminary
                quality checklist.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            """
            <div class="info-card">
                <h3>🧩 Skill Intelligence</h3>
                <p>Identify skills listed in your resume and organize
                them into categories.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            """
            <div class="info-card">
                <h3>🎯 Job Matching</h3>
                <p>Compare detected skills and text similarity against
                a target job description.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.stop()


# --------------------------------------------------
# EXTRACT RESUME TEXT
# --------------------------------------------------

try:
    with pymupdf.open(
        stream=uploaded_file.getvalue(),
        filetype="pdf",
    ) as document:
        resume_text = "\n".join(
            page.get_text() for page in document
        )

except Exception:
    st.error(
        "We could not read this PDF. Please try another "
        "text-based PDF resume."
    )
    st.stop()


if not resume_text.strip():
    st.warning(
        "No selectable text was found. This may be a scanned PDF. "
        "Try a text-based PDF."
    )
    st.stop()


try:
    resume_skills = detect_skills(resume_text)
    score, score_details, suggestions = score_resume(
        resume_text, resume_skills
    )
except Exception as error:
    st.error("Resume analysis could not be completed.")
    st.caption(str(error))
    st.stop()


# --------------------------------------------------
# JOB DESCRIPTION INPUT
# --------------------------------------------------

st.markdown("### Target Job")

with st.expander(
    "Paste a job description to compare your resume",
    expanded=True,
):
    with st.form("job_matching_form"):
        job_description = st.text_area(
            "Job description",
            height=160,
            placeholder=(
                "Paste the responsibilities, required skills, "
                "and qualifications from the job posting..."
            ),
        )

        submitted = st.form_submit_button(
            "Analyze Job Match",
            use_container_width=True,
        )

    if submitted:
        st.session_state["job_description"] = job_description.strip()


saved_job_description = st.session_state.get(
    "job_description", ""
)

job_skills = []
matched_skills = []
missing_skills = []
skill_coverage = None
tfidf_similarity = None

if saved_job_description:
    try:
        job_skills = detect_skills(saved_job_description)

        matched_skills = sorted(
            set(resume_skills) & set(job_skills),
            key=str.lower,
        )

        missing_skills = sorted(
            set(job_skills) - set(resume_skills),
            key=str.lower,
        )

        if job_skills:
            skill_coverage = round(
                len(matched_skills) / len(job_skills) * 100
            )

        try:
            tfidf_similarity = calculate_tfidf_similarity(
                resume_text,
                saved_job_description,
            )
        except ValueError:
            tfidf_similarity = 0

    except Exception as error:
        st.warning(f"Job matching could not be completed: {error}")


# --------------------------------------------------
# TOP METRIC CARDS
# --------------------------------------------------

st.markdown("### Your Resume at a Glance")

if skill_coverage is not None:
    metric3_value = f"{skill_coverage}%"
    metric3_label = "Required-Skill Coverage"
elif saved_job_description:
    metric3_value = "N/A"
    metric3_label = "No Recognized Job Skills"
else:
    metric3_value = "—"
    metric3_label = "Required-Skill Coverage"

metric1, metric2, metric3 = st.columns(3)

with metric1:
    st.metric(
        "Resume Quality",
        f"{score}/100",
        help="A preliminary rule-based checklist score.",
    )

with metric2:
    st.metric(
        "Skills Identified",
        len(resume_skills),
        help="Skills detected from the current skills list.",
    )

with metric3:
    st.metric(
        metric3_label,
        metric3_value,
        help="Percentage of detected job skills also found in the resume.",
    )

st.caption(
    "These indicators are preliminary. They are not official ATS "
    "scores or predictions of hiring outcomes."
)

st.divider()


# --------------------------------------------------
# DASHBOARD TABS
# --------------------------------------------------


overview_tab, skills_tab, matching_tab, improvement_tab, text_tab, market_tab = (
    st.tabs(
        [
            "📊 Overview",
            "🧩 Skills",
            "🎯 Job Match",
            "💡 Improvement Plan",
            "📄 Resume Text",
            "📈 Job-Market Trends",
        ],
        key="dashboard_tabs",
        on_change="rerun",
    )
)

# --------------------------------------------------
# OVERVIEW TAB
# --------------------------------------------------

with overview_tab:
    left_col, right_col = st.columns([1.15, 1])

    with left_col:
        st.markdown("### Resume Score Breakdown")

        score_df = pd.DataFrame(
            {
                "Category": list(score_details.keys()),
                "Points": list(score_details.values()),
            }
        )

        score_chart = (
            alt.Chart(score_df)
            .mark_bar(
                cornerRadiusEnd=5,
                color="#2457A7",
            )
            .encode(
                x=alt.X(
                    "Points:Q",
                    title="Points",
                    scale=alt.Scale(domain=[0, 30]),
                ),
                y=alt.Y(
                    "Category:N",
                    sort="-x",
                    title=None,
                ),
                tooltip=["Category", "Points"],
            )
            .properties(height=270)
        )

        st.altair_chart(
            score_chart,
            use_container_width=True,
        )

        st.caption(
            "Categories have different maximum possible points. "
            "The chart displays awarded points, not percentages."
        )

    with right_col:
        st.markdown("### Analysis Summary")

        st.markdown(
            f"""
            <div class="info-card">
                <div class="section-label">Extracted Content</div>
                <div class="metric-value">
                    {len(resume_text.split()):,}
                </div>
                <div class="metric-note">Words extracted from the PDF</div>
                <hr>
                <div class="section-label">Detected Skills</div>
                <div class="metric-value">{len(resume_skills)}</div>
                <div class="metric-note">
                    Matches from the configured skills list
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if saved_job_description:
            st.markdown("#### Job Comparison")

            if skill_coverage is not None:
                st.write(
                    f"**Required-skill coverage:** {skill_coverage}%"
                )
                st.progress(skill_coverage / 100)

            if tfidf_similarity is not None:
                st.write(
                    f"**TF-IDF text similarity:** {tfidf_similarity}%"
                )
                st.progress(tfidf_similarity / 100)

            st.caption(
                "These scores measure different things and should "
                "not be interpreted as a probability of getting hired."
            )
        else:
            st.info(
                "Add a job description above to unlock job-matching "
                "metrics and skill-gap results."
            )


# --------------------------------------------------
# SKILLS TAB
# --------------------------------------------------

with skills_tab:
    st.markdown("### Your Skill Profile")
    st.write(
        "Skills detected from your resume, grouped into broad "
        "categories for easier review."
    )

    grouped_skills = group_skills(resume_skills)

    if grouped_skills:
        for category, category_skills in grouped_skills.items():
            st.markdown(f"#### {category}")
            render_skill_tags(category_skills)
            st.write("")
    else:
        st.info(
            "No skills from the current list were found. "
            "Review data/skills.csv to expand the vocabulary."
        )

    with st.expander("View all detected skills"):
        st.write(", ".join(resume_skills) or "No skills detected.")


# --------------------------------------------------
# JOB MATCH TAB
# --------------------------------------------------

with matching_tab:
    st.markdown("### Job Fit Analysis")

    if not saved_job_description:
        st.info(
            "Paste a job description in the section above and click "
            "'Analyze Job Match' to see your results."
        )

    else:
        if skill_coverage is not None:
            match_col1, match_col2 = st.columns(2)

            with match_col1:
                st.markdown(
                    f"""
                    <div class="info-card">
                        <div class="section-label">
                            Required-Skill Coverage
                        </div>
                        <div class="metric-value">
                            {skill_coverage}%
                        </div>
                        <div class="metric-note">
                            {len(matched_skills)} of
                            {len(job_skills)} recognized job skills
                            appear in the resume
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with match_col2:
                st.markdown(
                    f"""
                    <div class="info-card">
                        <div class="section-label">
                            TF-IDF Text Similarity
                        </div>
                        <div class="metric-value">
                            {tfidf_similarity}%
                        </div>
                        <div class="metric-note">
                            Similarity of weighted words and phrases
                            in the two documents
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            st.caption(
                "Skill coverage and TF-IDF similarity use different "
                "methods. Neither is an official job-fit or ATS score."
            )

        else:
            st.warning(
                "No skills from the current skills list were found "
                "in this job description. Add more relevant terms "
                "to data/skills.csv if needed."
            )

            if tfidf_similarity is not None:
                st.metric(
                    "TF-IDF Text Similarity",
                    f"{tfidf_similarity}%",
                )

        st.divider()

        result_col1, result_col2 = st.columns(2)

        with result_col1:
            st.markdown("#### ✓ Matching Skills")

            if matched_skills:
                render_skill_tags(
                    matched_skills,
                    css_class="matched-tag",
                )
            else:
                st.write("No matching skills detected.")

        with result_col2:
            st.markdown("#### + Skills to Develop")

            if missing_skills:
                render_skill_tags(
                    missing_skills,
                    css_class="missing-tag",
                )
            elif job_skills:
                st.success(
                    "All recognized job-description skills were "
                    "detected in the resume."
                )
            else:
                st.write(
                    "No recognized skills were available for comparison."
                )

        st.divider()

        st.markdown("#### Requirements Detected")
        st.write(
            ", ".join(job_skills)
            if job_skills
            else "No matching skills were found in the job description."
        )

        st.caption(
            "A skill detected in a resume is not proof of proficiency. "
            "Missing skills may also reflect limitations in the "
            "configured vocabulary or PDF text extraction."
        )


# --------------------------------------------------
# IMPROVEMENT PLAN TAB
# --------------------------------------------------

with improvement_tab:
    st.markdown("### Resume Improvement Plan")

    st.write(
        "Use these checklist suggestions to review your resume. "
        "Only include skills and achievements that are accurate."
    )

    if suggestions:
        for index, suggestion in enumerate(suggestions, start=1):
            st.markdown(
                f"""
                <div class="info-card">
                    <span class="section-label">
                        RECOMMENDATION {index:02d}
                    </span>
                    <p style="color:#263653; font-weight:600;
                    margin-top:8px; margin-bottom:0;">
                        {suggestion}
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )
    else:
        st.success(
            "No checklist suggestions were triggered. "
            "Review your resume against the target role as well."
        )

    if saved_job_description and missing_skills:
        st.markdown("### Job-Specific Learning Priorities")

        st.write(
            "These skills appeared in the job description but were "
            "not detected in the resume. Verify your actual knowledge "
            "before deciding what to learn."
        )

        for skill in missing_skills:
            st.markdown(f"- **{skill}** — review the fundamentals and "
                        "consider a relevant practice project.")

    elif saved_job_description and job_skills:
        st.success(
            "All recognized job skills were found in the resume. "
            "Review the quality of your project evidence and experience."
        )

    st.warning(
        "This tool does not verify whether you possess a skill. "
        "Do not add a skill or achievement to your resume unless it "
        "truthfully reflects your experience."
    )


# --------------------------------------------------
# EXTRACTED RESUME TEXT TAB
# --------------------------------------------------

with text_tab:
    st.markdown("### Extracted Resume Content")

    st.write(
        "Check the extracted text to confirm that your PDF was read "
        "correctly. Formatting and visual design are not assessed "
        "by the current extraction process."
    )

    st.text_area(
        "Resume text",
        value=resume_text,
        height=420,
        disabled=True,
    )

    st.download_button(
        label="Download Extracted Text",
        data=resume_text,
        file_name="extracted_resume.txt",
        mime="text/plain",
        use_container_width=True,
    )


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "ResumeIQ · Educational prototype · "
    "Results depend on extracted PDF text and the configured skills list."
)
