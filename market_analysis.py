
import pandas as pd


def load_job_postings(path="data/job_postings.csv"):
    df = pd.read_csv(path)

    required = {"job_id", "job_role", "skills", "posted_date"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df["skills"] = df["skills"].fillna("")
    return df


def calculate_skill_demand(df, role=None):
    if role and role != "All Roles":
        df = df[df["job_role"] == role]

    total_jobs = len(df)
    counts = {}

    for skills_text in df["skills"]:
        # Count each skill once per job posting.
        unique_skills = {
            skill.strip().casefold(): skill.strip()
            for skill in skills_text.split(";")
            if skill.strip()
        }

        for key, name in unique_skills.items():
            if key not in counts:
                counts[key] = {"skill": name, "job_count": 0}
            counts[key]["job_count"] += 1

    if not counts:
        return pd.DataFrame(
            columns=["skill", "job_count", "demand_percent"]
        )

    result = pd.DataFrame(counts.values())
    result["demand_percent"] = (
        result["job_count"] / total_jobs * 100
    ).round(1)

    return result.sort_values(
        ["job_count", "skill"],
        ascending=[False, True]
    ).reset_index(drop=True)


def compare_job_roles(df):
    results = []

    for role in sorted(df["job_role"].dropna().unique()):
        role_data = calculate_skill_demand(df, role)

        for _, item in role_data.iterrows():
            results.append({
                "job_role": role,
                "skill": item["skill"],
                "job_count": int(item["job_count"]),
                "demand_percent": float(item["demand_percent"])
            })

    return pd.DataFrame(results)


if __name__ == "__main__":
    jobs = load_job_postings()

    print("\n=== TOP 10 SKILLS ACROSS ALL ROLES ===")
    print(
        calculate_skill_demand(jobs)
        .head(10)
        .to_string(index=False)
    )

    print("\n=== SKILL DEMAND BY ROLE ===")
    print(compare_job_roles(jobs).to_string(index=False))
