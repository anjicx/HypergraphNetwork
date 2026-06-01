import pandas as pd
import numpy as np

male_hyperedge_table = pd.read_csv(
    "male_hyperedge_table.csv",
    parse_dates=["first_date"]
)

female_hyperedge_table = pd.read_csv(
    "female_hyperedge_table.csv",
    parse_dates=["first_date"]
)

hypergraphs = {
    "Male": male_hyperedge_table,
    "Female": female_hyperedge_table
}

"""STRUCTURAL STATISTICS"""

def hypergraph_statistics(hyperedge_table):
    unique_memberships = hyperedge_table.drop_duplicates(
        ["hyperedge_index", "node_id"]
    ).copy()

    n_hyperedges = unique_memberships["hyperedge_index"].nunique()
    n_nodes = unique_memberships["node_id"].nunique()
    n_memberships = len(unique_memberships)

    edge_sizes = (
        unique_memberships
        .groupby("hyperedge_index")["node_id"]
        .nunique()
        .rename("n_unique_diagnoses")
    )

    node_degrees = (
        unique_memberships
        .groupby("node_id")["hyperedge_index"]
        .nunique()
        .rename("degree_n_patients")
    )

    density = n_memberships / (n_hyperedges * n_nodes)
    sparsity = 1 - density

    stats = pd.DataFrame({
        "n_hyperedges_patients": [n_hyperedges],
        "n_nodes_diagnoses": [n_nodes],
        "n_memberships_patient_diagnosis": [n_memberships],

        "incidence_density": [density],
        "incidence_sparsity": [sparsity],

        "avg_hyperedge_size": [edge_sizes.mean()],
        "median_hyperedge_size": [edge_sizes.median()],
        "min_hyperedge_size": [edge_sizes.min()],
        "max_hyperedge_size": [edge_sizes.max()],

        "avg_node_degree": [node_degrees.mean()],
        "median_node_degree": [node_degrees.median()],
        "min_node_degree": [node_degrees.min()],
        "max_node_degree": [node_degrees.max()]
    })

    return stats, edge_sizes, node_degrees

def top_diagnoses(hyperedge_table, top_n=50):
    unique_memberships = hyperedge_table.drop_duplicates(
        ["hyperedge_index", "node_id"]
    ).copy()

    top_diag = (
        unique_memberships
        .groupby(["node_id", "diagnose_id", "icd_code", "descr"])["hyperedge_index"]
        .nunique()
        .reset_index(name="n_patients")
        .sort_values("n_patients", ascending=False)
        .head(top_n)
    )

    n_patients = unique_memberships["hyperedge_index"].nunique()
    top_diag["prevalence"] = top_diag["n_patients"] / n_patients

    return top_diag

# patient burden = number of unique diagnoses per patient

def hyperedge_size_distribution(hyperedge_table):
    unique_memberships = hyperedge_table.drop_duplicates(
        ["hyperedge_index", "node_id"]
    ).copy()

    edge_sizes = (
        unique_memberships
        .groupby("hyperedge_index")["node_id"]
        .nunique()
        .rename("n_unique_diagnoses")
    )

    summary = (
        edge_sizes
        .describe(percentiles=[0.25, 0.50, 0.75, 0.90, 0.95, 0.99])
        .to_frame()
        .T
    )

    burden_groups = pd.DataFrame({
        "patients_total": [len(edge_sizes)],

        "patients_ge_3_diag": [(edge_sizes >= 3).sum()],
        "patients_ge_5_diag": [(edge_sizes >= 5).sum()],
        "patients_ge_10_diag": [(edge_sizes >= 10).sum()],
        "patients_ge_20_diag": [(edge_sizes >= 20).sum()],
        "patients_ge_50_diag": [(edge_sizes >= 50).sum()],

        "pct_ge_3_diag": [(edge_sizes >= 3).mean()],
        "pct_ge_5_diag": [(edge_sizes >= 5).mean()],
        "pct_ge_10_diag": [(edge_sizes >= 10).mean()],
        "pct_ge_20_diag": [(edge_sizes >= 20).mean()],
        "pct_ge_50_diag": [(edge_sizes >= 50).mean()]
    })

    edge_size_table = edge_sizes.reset_index()

    return summary, burden_groups, edge_size_table

# diagnosis degree = number of patients with that diagnosis


def node_degree_distribution(hyperedge_table):
    unique_memberships = hyperedge_table.drop_duplicates(
        ["hyperedge_index", "node_id"]
    ).copy()

    node_degrees = (
        unique_memberships
        .groupby("node_id")["hyperedge_index"]
        .nunique()
        .rename("degree_n_patients")
    )

    node_info = (
        hyperedge_table[["node_id", "diagnose_id", "icd_code", "descr"]]
        .drop_duplicates("node_id")
        .set_index("node_id")
    )

    node_degree_table = node_info.join(node_degrees, how="inner").reset_index()

    n_patients = unique_memberships["hyperedge_index"].nunique()
    node_degree_table["prevalence"] = (
        node_degree_table["degree_n_patients"] / n_patients
    )

    summary = (
        node_degrees
        .describe(percentiles=[0.25, 0.50, 0.75, 0.90, 0.95, 0.99])
        .to_frame()
        .T
    )

    rare_stats = pd.DataFrame({
        "n_nodes_total": [len(node_degrees)],

        "n_degree_1": [(node_degrees == 1).sum()],
        "n_degree_le_5": [(node_degrees <= 5).sum()],
        "n_degree_le_10": [(node_degrees <= 10).sum()],
        "n_degree_le_20": [(node_degrees <= 20).sum()],

        "pct_degree_1": [(node_degrees == 1).mean()],
        "pct_degree_le_5": [(node_degrees <= 5).mean()],
        "pct_degree_le_10": [(node_degrees <= 10).mean()],
        "pct_degree_le_20": [(node_degrees <= 20).mean()]
    })

    top_nodes = (
        node_degree_table
        .sort_values("degree_n_patients", ascending=False)
        .head(50)
    )

    return summary, rare_stats, top_nodes, node_degree_table

"""#TEMPORAL SEQUENCE STATISTICS"""

#TEMPORAL SEQUENCE STATISTICS
# sequence length, follow-up length, diagnoses per year

def temporal_sequence_statistics(hyperedge_table):
    df = hyperedge_table.drop_duplicates(
        ["patient_no", "node_id", "first_date"]
    ).copy()

    df["first_date"] = pd.to_datetime(df["first_date"])

    patient_temporal = (
        df.groupby("patient_no")
        .agg(
            sequence_length=("node_id", "nunique"),
            first_diagnosis_date=("first_date", "min"),
            last_diagnosis_date=("first_date", "max"),
            n_unique_dates=("first_date", "nunique")
        )
        .reset_index()
    )

    patient_temporal["follow_up_days"] = (
        patient_temporal["last_diagnosis_date"]
        - patient_temporal["first_diagnosis_date"]
    ).dt.days

    patient_temporal["follow_up_years"] = (
        patient_temporal["follow_up_days"] / 365.25
    )

    patient_temporal["diagnoses_per_year"] = np.where(
        patient_temporal["follow_up_years"] > 0,
        patient_temporal["sequence_length"] / patient_temporal["follow_up_years"],
        np.nan
    )

    summary = (
        patient_temporal[
            [
                "sequence_length",
                "n_unique_dates",
                "follow_up_days",
                "follow_up_years",
                "diagnoses_per_year"
            ]
        ]
        .describe(percentiles=[0.25, 0.50, 0.75, 0.90, 0.95, 0.99])
    )

    return summary, patient_temporal

# SAME-DAY DIAGNOSIS TIES

def same_day_tie_statistics(hyperedge_table):
    df = hyperedge_table.drop_duplicates(
        ["patient_no", "node_id", "first_date"]
    ).copy()

    df["first_date"] = pd.to_datetime(df["first_date"])

    same_day_counts = (
        df.groupby(["patient_no", "first_date"])["node_id"]
        .nunique()
        .reset_index(name="n_diagnoses_same_day")
    )

    patient_tie_stats = (
        same_day_counts.groupby("patient_no")
        .agg(
            max_diagnoses_same_day=("n_diagnoses_same_day", "max"),
            n_dates_with_multiple_diagnoses=(
                "n_diagnoses_same_day",
                lambda x: (x > 1).sum()
            ),
            n_unique_dates=("first_date", "nunique")
        )
        .reset_index()
    )

    patient_tie_stats["has_same_day_tie"] = (
        patient_tie_stats["n_dates_with_multiple_diagnoses"] > 0
    )

    summary = pd.DataFrame({
        "n_patients": [patient_tie_stats["patient_no"].nunique()],
        "n_patients_with_same_day_tie": [
            patient_tie_stats["has_same_day_tie"].sum()
        ],
        "pct_patients_with_same_day_tie": [
            patient_tie_stats["has_same_day_tie"].mean()
        ],
        "avg_max_diagnoses_same_day": [
            patient_tie_stats["max_diagnoses_same_day"].mean()
        ],
        "max_diagnoses_same_day": [
            patient_tie_stats["max_diagnoses_same_day"].max()
        ]
    })

    return summary, patient_tie_stats, same_day_counts

# A -> B, B -> C, C -> D like 1 step

def consecutive_transition_statistics(hyperedge_table):
    df = (
        hyperedge_table
        .drop_duplicates(["patient_no", "node_id", "first_date"])
        .sort_values(["patient_no", "first_date", "node_id"])
        .copy()
    )

    df["first_date"] = pd.to_datetime(df["first_date"])

    df["next_node_id"] = df.groupby("patient_no")["node_id"].shift(-1)
    df["next_first_date"] = df.groupby("patient_no")["first_date"].shift(-1)

    transitions = df.dropna(subset=["next_node_id"]).copy()
    transitions["next_node_id"] = transitions["next_node_id"].astype(int)

    transitions["lag_days"] = (
        transitions["next_first_date"] - transitions["first_date"]
    ).dt.days

    transition_counts = (
        transitions
        .groupby(["node_id", "next_node_id"])
        .agg(
            n_patients=("patient_no", "nunique"),
            n_transitions=("patient_no", "size"),
            median_lag_days=("lag_days", "median"),
            mean_lag_days=("lag_days", "mean"),
            min_lag_days=("lag_days", "min"),
            max_lag_days=("lag_days", "max"),
            pct_zero_lag=("lag_days", lambda x: (x == 0).mean())
        )
        .reset_index()
        .sort_values(["n_patients", "n_transitions"], ascending=False)
    )

    source_info = (
        hyperedge_table[["node_id", "icd_code", "descr"]]
        .drop_duplicates("node_id")
        .rename(columns={
            "node_id": "source_node_id",
            "icd_code": "source_icd",
            "descr": "source_descr"
        })
    )

    target_info = (
        hyperedge_table[["node_id", "icd_code", "descr"]]
        .drop_duplicates("node_id")
        .rename(columns={
            "node_id": "target_node_id",
            "icd_code": "target_icd",
            "descr": "target_descr"
        })
    )

    transition_counts = transition_counts.rename(columns={
        "node_id": "source_node_id",
        "next_node_id": "target_node_id"
    })

    transition_counts = transition_counts.merge(
        source_info,
        on="source_node_id",
        how="left"
    )

    transition_counts = transition_counts.merge(
        target_info,
        on="target_node_id",
        how="left"
    )

    lag_summary = (
        transitions["lag_days"]
        .describe(percentiles=[0.25, 0.50, 0.75, 0.90, 0.95, 0.99])
        .to_frame()
        .T
    )

    lag_quality = pd.DataFrame({
        "n_consecutive_transitions": [len(transitions)],
        "n_negative_lags": [(transitions["lag_days"] < 0).sum()],
        "n_zero_lags": [(transitions["lag_days"] == 0).sum()],
        "pct_zero_lags": [(transitions["lag_days"] == 0).mean()],
        "median_lag_days": [transitions["lag_days"].median()],
        "mean_lag_days": [transitions["lag_days"].mean()]
    })

    return transition_counts, transitions, lag_summary, lag_quality

# PREFIX PREDICTION STATISTICS
# for next-disease prediction:
# {A} -> B
# {A, B} -> C
# {A, B, C} -> D

def prefix_prediction_statistics(hyperedge_table, save_prefix_rows=False):
    df = (
        hyperedge_table
        .drop_duplicates(["patient_no", "node_id", "first_date"])
        .sort_values(["patient_no", "first_date", "node_id"])
        .copy()
    )

    df["first_date"] = pd.to_datetime(df["first_date"])

    rows = []

    for patient_no, g in df.groupby("patient_no"):
        g = g.sort_values(["first_date", "node_id"]).reset_index(drop=True)

        sequence = g["node_id"].tolist()
        dates = g["first_date"].tolist()

        if len(sequence) < 2:
            continue

        for t in range(1, len(sequence)):
            prefix = sequence[:t]
            target = sequence[t]

            rows.append({
                "patient_no": patient_no,
                "prefix_length": len(prefix),
                "target_node_id": target,
                "target_date": dates[t],
                "previous_date": dates[t - 1],
                "lag_days_from_previous": (dates[t] - dates[t - 1]).days
            })

    prefix_df = pd.DataFrame(rows)

    if prefix_df.empty:
        if save_prefix_rows:
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    prefix_summary = (
        prefix_df[["prefix_length", "lag_days_from_previous"]]
        .describe(percentiles=[0.25, 0.50, 0.75, 0.90, 0.95, 0.99])
    )

    target_distribution = (
        prefix_df
        .groupby("target_node_id")
        .agg(
            n_target_occurrences=("patient_no", "size"),
            n_patients=("patient_no", "nunique")
        )
        .reset_index()
        .sort_values("n_target_occurrences", ascending=False)
    )

    target_info = (
        hyperedge_table[["node_id", "diagnose_id", "icd_code", "descr"]]
        .drop_duplicates("node_id")
        .rename(columns={"node_id": "target_node_id"})
    )

    target_distribution = target_distribution.merge(
        target_info,
        on="target_node_id",
        how="left"
    )

    target_imbalance_summary = pd.DataFrame({
        "n_prefix_samples": [len(prefix_df)],
        "n_unique_target_diagnoses": [prefix_df["target_node_id"].nunique()],
        "most_common_target_count": [
            target_distribution["n_target_occurrences"].max()
        ],
        "most_common_target_share": [
            target_distribution["n_target_occurrences"].max() / len(prefix_df)
        ],
        "n_targets_with_1_sample": [
            (target_distribution["n_target_occurrences"] == 1).sum()
        ],
        "pct_targets_with_1_sample": [
            (target_distribution["n_target_occurrences"] == 1).mean()
        ]
    })

    if save_prefix_rows:
        return prefix_summary, target_distribution, target_imbalance_summary, prefix_df

    return prefix_summary, target_distribution, target_imbalance_summary

"""for running all statistics"""

all_basic_stats = []
all_hyperedge_size_summary = []
all_burden_groups = []
all_node_degree_summary = []
all_rare_stats = []
all_temporal_summary = []
all_same_day_summary = []
all_lag_summary = []
all_lag_quality = []
all_prefix_summary = []
all_target_imbalance = []

for group_name, table in hypergraphs.items():
    print(f"\nProcessing {group_name} hypergraph...")

    group_prefix = group_name.lower()

    # 1. Basic hypergraph statistics
    stats, edge_sizes, node_degrees = hypergraph_statistics(table)
    stats.insert(0, "group", group_name)
    all_basic_stats.append(stats)

    edge_sizes.reset_index().to_csv(
        f"{group_prefix}_hyperedge_sizes_per_patient.csv",
        index=False
    )

    node_degrees.reset_index().to_csv(
        f"{group_prefix}_node_degrees_raw.csv",
        index=False
    )

    # 2. Top diagnoses
    top_diag = top_diagnoses(table, top_n=50)
    top_diag.to_csv(
        f"{group_prefix}_top_50_diagnoses.csv",
        index=False
    )

    # 3. Hyperedge size distribution
    size_summary, burden_groups, edge_size_table = hyperedge_size_distribution(table)

    size_summary.insert(0, "group", group_name)
    burden_groups.insert(0, "group", group_name)

    all_hyperedge_size_summary.append(size_summary)
    all_burden_groups.append(burden_groups)

    edge_size_table.to_csv(
        f"{group_prefix}_hyperedge_size_table.csv",
        index=False
    )

    # 4. Node degree distribution
    node_summary, rare_stats, top_nodes, node_degree_table = node_degree_distribution(table)

    node_summary.insert(0, "group", group_name)
    rare_stats.insert(0, "group", group_name)

    all_node_degree_summary.append(node_summary)
    all_rare_stats.append(rare_stats)

    top_nodes.to_csv(
        f"{group_prefix}_top_50_nodes_by_degree.csv",
        index=False
    )

    node_degree_table.to_csv(
        f"{group_prefix}_node_degree_prevalence_table.csv",
        index=False
    )

    # 5. Temporal sequence statistics
    temporal_summary, patient_temporal = temporal_sequence_statistics(table)

    temporal_summary.insert(0, "metric", temporal_summary.index)
    temporal_summary.insert(0, "group", group_name)
    temporal_summary = temporal_summary.reset_index(drop=True)

    all_temporal_summary.append(temporal_summary)

    patient_temporal.to_csv(
        f"{group_prefix}_patient_temporal_summary.csv",
        index=False
    )

    # 6. Same-day ties
    same_day_summary, patient_tie_stats, same_day_counts = same_day_tie_statistics(table)

    same_day_summary.insert(0, "group", group_name)
    all_same_day_summary.append(same_day_summary)

    patient_tie_stats.to_csv(
        f"{group_prefix}_patient_same_day_tie_stats.csv",
        index=False
    )

    same_day_counts.to_csv(
        f"{group_prefix}_same_day_counts.csv",
        index=False
    )

    # 7. Consecutive transition statistics
    transition_counts, transitions, lag_summary, lag_quality = consecutive_transition_statistics(table)

    lag_summary.insert(0, "group", group_name)
    lag_quality.insert(0, "group", group_name)

    all_lag_summary.append(lag_summary)
    all_lag_quality.append(lag_quality)

    transition_counts.to_csv(
        f"{group_prefix}_consecutive_transition_counts_full.csv",
        index=False
    )

    transition_counts.head(50).to_csv(
        f"{group_prefix}_top_50_consecutive_transitions.csv",
        index=False
    )

    transitions.to_csv(
        f"{group_prefix}_consecutive_transition_rows.csv",
        index=False
    )

    # 8. Prefix prediction statistics
    prefix_summary, target_distribution, target_imbalance_summary, prefix_df = (
        prefix_prediction_statistics(table, save_prefix_rows=True)
    )

    prefix_summary.insert(0, "metric", prefix_summary.index)
    prefix_summary.insert(0, "group", group_name)
    prefix_summary = prefix_summary.reset_index(drop=True)

    target_imbalance_summary.insert(0, "group", group_name)

    all_prefix_summary.append(prefix_summary)
    all_target_imbalance.append(target_imbalance_summary)

    target_distribution.to_csv(
        f"{group_prefix}_target_distribution_for_next_disease.csv",
        index=False
    )

    prefix_df.to_csv(
        f"{group_prefix}_prefix_prediction_samples.csv",
        index=False
    )

    print(f"Finished {group_name}. Files saved with prefix: {group_prefix}_")

""" SAVE STATISTICS"""

pd.concat(all_basic_stats, ignore_index=True).to_csv(
    "combined_basic_hypergraph_statistics.csv",
    index=False
)

pd.concat(all_hyperedge_size_summary, ignore_index=True).to_csv(
    "combined_hyperedge_size_summary.csv",
    index=False
)

pd.concat(all_burden_groups, ignore_index=True).to_csv(
    "combined_multimorbidity_burden_groups.csv",
    index=False
)

pd.concat(all_node_degree_summary, ignore_index=True).to_csv(
    "combined_node_degree_summary.csv",
    index=False
)

pd.concat(all_rare_stats, ignore_index=True).to_csv(
    "combined_rare_diagnosis_stats.csv",
    index=False
)

pd.concat(all_temporal_summary, ignore_index=True).to_csv(
    "combined_temporal_sequence_summary.csv",
    index=False
)

pd.concat(all_same_day_summary, ignore_index=True).to_csv(
    "combined_same_day_tie_summary.csv",
    index=False
)

pd.concat(all_lag_summary, ignore_index=True).to_csv(
    "combined_consecutive_lag_summary.csv",
    index=False
)

pd.concat(all_lag_quality, ignore_index=True).to_csv(
    "combined_lag_quality_summary.csv",
    index=False
)

pd.concat(all_prefix_summary, ignore_index=True).to_csv(
    "combined_prefix_prediction_summary.csv",
    index=False
)

pd.concat(all_target_imbalance, ignore_index=True).to_csv(
    "combined_target_imbalance_summary.csv",
    index=False
)

print("All network/hypergraph statistics saved in the current working directory.")

# sex x age group

sex_age_rows = []
patient_burden_rows = []

for sex, table in hypergraphs.items():
    table = table.copy()
    table["age_group"] = table["age_group"].astype(str).str.strip()

    for age_group, subtable in table.groupby("age_group"):
        subtable = subtable.copy()

        if subtable.empty:
            continue

        # Basic hypergraph stats
        stats, edge_sizes, node_degrees = hypergraph_statistics(subtable)

        # Burden groups
        size_summary, burden_groups, edge_size_table = hyperedge_size_distribution(subtable)

        # Temporal stats
        temporal_summary, patient_temporal = temporal_sequence_statistics(subtable)

        # Prefix prediction stats
        prefix_summary, target_distribution, target_imbalance_summary = (
            prefix_prediction_statistics(subtable, save_prefix_rows=False)
        )

        # Some age groups may have too few samples for prefix prediction
        if target_imbalance_summary.empty:
            n_prefix_samples = 0
            n_unique_target_diagnoses = 0
            most_common_target_share = np.nan
            pct_targets_with_1_sample = np.nan
        else:
            n_prefix_samples = target_imbalance_summary["n_prefix_samples"].iloc[0]
            n_unique_target_diagnoses = target_imbalance_summary["n_unique_target_diagnoses"].iloc[0]
            most_common_target_share = target_imbalance_summary["most_common_target_share"].iloc[0]
            pct_targets_with_1_sample = target_imbalance_summary["pct_targets_with_1_sample"].iloc[0]

        row = {
            "sex": sex,
            "age_group": age_group,

            "n_hyperedges_patients": stats["n_hyperedges_patients"].iloc[0],
            "n_nodes_diagnoses": stats["n_nodes_diagnoses"].iloc[0],
            "n_memberships_patient_diagnosis": stats["n_memberships_patient_diagnosis"].iloc[0],

            "incidence_density": stats["incidence_density"].iloc[0],
            "incidence_sparsity": stats["incidence_sparsity"].iloc[0],

            "median_hyperedge_size": stats["median_hyperedge_size"].iloc[0],
            "max_hyperedge_size": stats["max_hyperedge_size"].iloc[0],
            "median_node_degree": stats["median_node_degree"].iloc[0],
            "max_node_degree": stats["max_node_degree"].iloc[0],

            "pct_ge_5_diag": burden_groups["pct_ge_5_diag"].iloc[0],
            "pct_ge_10_diag": burden_groups["pct_ge_10_diag"].iloc[0],

            "median_sequence_length": patient_temporal["sequence_length"].median(),
            "median_follow_up_years": patient_temporal["follow_up_years"].median(),
            "median_diagnoses_per_year": patient_temporal["diagnoses_per_year"].median(),

            "n_prefix_samples": n_prefix_samples,
            "n_unique_target_diagnoses": n_unique_target_diagnoses,
            "most_common_target_share": most_common_target_share,
            "pct_targets_with_1_sample": pct_targets_with_1_sample,
        }

        sex_age_rows.append(row)

        # Patient-level burden table for plots
        burden_age = edge_size_table.copy()
        burden_age["sex"] = sex
        burden_age["age_group"] = age_group

        patient_burden_rows.append(burden_age)

sex_age_summary = pd.DataFrame(sex_age_rows)
patient_burden_age = pd.concat(patient_burden_rows, ignore_index=True)

# Sort age groups correctly
sex_age_summary["age_start"] = (
    sex_age_summary["age_group"]
    .astype(str)
    .str.extract(r"(\d+)")
    .astype(float)
)

patient_burden_age["age_start"] = (
    patient_burden_age["age_group"]
    .astype(str)
    .str.extract(r"(\d+)")
    .astype(float)
)

sex_age_summary = sex_age_summary.sort_values(["sex", "age_start"])
patient_burden_age = patient_burden_age.sort_values(["sex", "age_start"])

sex_age_summary.to_csv(
    "KEY_SUMMARY_BY_SEX_AGE_FOR_REPORT.csv",
    index=False
)

patient_burden_age.to_csv(
    "PATIENT_BURDEN_BY_SEX_AGE_FOR_PLOTS.csv",
    index=False
)

print("Saved sex-age summary to: KEY_SUMMARY_BY_SEX_AGE_FOR_REPORT.csv")
print("Saved patient burden by sex-age to: PATIENT_BURDEN_BY_SEX_AGE_FOR_PLOTS.csv")

