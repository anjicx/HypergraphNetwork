import os
import pandas as pd
import pandas as pd

# loading the data
diagnosis = pd.read_csv("data/table_diagnoses.csv", sep=";")
stays = pd.read_csv("data/final_one_percent_stays.csv", sep=";")
stays_secondaries = pd.read_csv("data/final_one_percent_stays_secondaries.csv", sep=";")
age = pd.read_csv("data/table_age.csv", sep=";", encoding="latin1")

"""1. PATIENT FILTERING
Making a cohort of patients who have at least 3 unique diagnosis and their first appearance is from 20 age
"""

#Patient stays filtered
stays = stays.merge(age, on="ag_id", how="left")
stays = stays.rename(columns={"age": "age_group"})

patient_stays = stays[["patient_no", "entry_date", "exit_date", "sex_id", "age_group"]].copy()
patient_stays = patient_stays[patient_stays["sex_id"].isin([1, 2])].copy()

patient_stays["entry_date"] = pd.to_datetime(patient_stays["entry_date"])
patient_stays["exit_date"] = pd.to_datetime(patient_stays["exit_date"])
patient_stays = patient_stays.sort_values(["patient_no", "entry_date"])#patient stays sorted by entry date


#just takes the year from the entry date
patient_stays["year"] = patient_stays["entry_date"].dt.year
#patient_stays
#COUNTNING NUMBER OF UNIQUE YEARS IN VISITS OF EACH PATIENT
unique_years = (
    patient_stays.groupby("patient_no")["year"].nunique()# group by patient, count distinct years
    .reset_index(name="n_unique_years")
)

  # assign age group and sex from the first visit
first_visit = patient_stays.drop_duplicates("patient_no", keep="first")[
    ["patient_no", "sex_id", "age_group"]
]

patient_filter = first_visit.merge(unique_years, on="patient_no", how="left")

patient_filter["sex"] = patient_filter["sex_id"].map({
    1: "Male",
    2: "Female"
})

# FIRST CONDITION:20+ AGE GROUP
# age group 20+. Counting first age group in the dataset-to be 20+? 40 bis 44 Jahre lake this is age_group
patient_filter["age_group"] = patient_filter["age_group"].astype(str).str.strip()
patient_filter["age_start"] = (
    patient_filter["age_group"]
    .str.extract(r"(\d+)")
    .astype(float)
)# to extract the start age in the age group
patient_filter = patient_filter[patient_filter["age_start"] >= 20].copy()#filter 20+

# SECOND CONDITION NUMBER OF UNIQUE YEARS FOR THAT PATIENT MORE THEN 3
patient_filter = patient_filter[patient_filter["n_unique_years"] >= 3].copy()
#ALL THE PATIENTS THAT CAME THROUGH FILTER

#unique_years.head(5)
#first_visit.head(5)

"""patient_filter is our cohort of patients -> filtering only the stays from our cohort"""

filtered_stays = stays[stays["patient_no"].isin(patient_filter["patient_no"])].copy()
filtered_stays["entry_date"] = pd.to_datetime(filtered_stays["entry_date"])
filtered_stays["exit_date"] = pd.to_datetime(filtered_stays["exit_date"], errors="coerce")

print("Filtered stays shape:", filtered_stays.shape)

"""2. Diagnosis nodes

Creating nodes of diagnosis(dictionary containing all diagnosis), and table of stays and connected diagnosis to that stay.Each row is one diagnosis connected to that stay
"""

# building nodes of diagnosis

nodes = diagnosis.copy()
nodes["diagnose_id"] = pd.to_numeric(nodes["diagnose_id"])
nodes["node_id"] = range(len(nodes))
nodes = nodes[["node_id", "diagnose_id", "descr", "icd_code"]].copy()



#creating a tbl of primary diagnosis

primary = filtered_stays[["stay_id", "patient_no", "entry_date", "pri_diag_id"]].copy()#stays has for key to diagnosis
primary = primary.rename(columns={"pri_diag_id": "diagnose_id"})#rename foreign key diagnose_id
primary["diagnose_id"] = pd.to_numeric(primary["diagnose_id"])
primary["role"] = "primary"



#creating a tbl of secondary diagnosis

filtered_stays_secondaries = stays_secondaries[stays_secondaries["stay_id"].isin(filtered_stays["stay_id"])].copy()
filtered_stays_secondaries["sec_diag_id"] = pd.to_numeric(filtered_stays_secondaries["sec_diag_id"])
secondary = filtered_stays_secondaries.merge(filtered_stays[["stay_id", "patient_no", "entry_date"]],on="stay_id",how="inner")
secondary = secondary.rename(columns={"sec_diag_id": "diagnose_id"})
secondary["role"] = "secondary"



# NODES TABLE: node id connected to diagnosis
stay_diagnoses = pd.concat([primary, secondary], ignore_index=True)
# remove rows with missing diagnosis ids
stay_diagnoses = stay_diagnoses.dropna(subset=["diagnose_id"]).copy()
# inner join
stay_diagnoses = pd.merge(stay_diagnoses,nodes,on="diagnose_id",how="inner")

#stay_diagnoses.head(5)

"""For each patient keep only the first appearance of diagnosis.Then merge the whole data and sort so you can track which diagnosis was first.Afterwards divide by gender."""

patient_first_diag = (stay_diagnoses.groupby(["patient_no", "node_id"], as_index=False)["entry_date"].min()
.rename(columns={"entry_date": "first_date"})
)
patient_first_diag = patient_first_diag.merge(
    nodes[["node_id", "diagnose_id", "descr", "icd_code"]].drop_duplicates(subset=["node_id"]),
    on="node_id",
    how="left")
# sort for each patient diagnosis order
patient_first_diag = patient_first_diag.sort_values(
    ["patient_no", "first_date", "node_id"]
).copy()

patient_info = patient_filter[
    ["patient_no", "sex_id", "sex", "age_group"]
].drop_duplicates().copy()

patient_first_diag_sex_age = patient_first_diag.merge(
    patient_info,
    on="patient_no",
    how="left"
)

# sorted for diagnosis timestamp order, unique diagnosis per gender
male_diag = patient_first_diag_sex_age[
    patient_first_diag_sex_age["sex"] == "Male"
].copy()

female_diag = patient_first_diag_sex_age[
    patient_first_diag_sex_age["sex"] == "Female"
].copy()

print("Male rows:", male_diag.shape)
print("Female rows:", female_diag.shape)
male_diag.head(5)

"""Instead of incidence matrix the hyperedge table is sotred, where for each patient from the previous table we add hyperedge index."""

def build_hypergraph_tables(df):
    df = df.sort_values(["patient_no", "first_date", "node_id"]).copy()

    # One hyperedge = one patient
    df["hyperedge_index"] = df["patient_no"]

    # Hyperedge table: patient-diagnosis memberships
    hyperedge_table = df[
        [
            "hyperedge_index",
            "patient_no",
            "sex_id",
            "sex",
            "age_group",
            "node_id",
            "diagnose_id",
            "descr",
            "icd_code",
            "first_date"
        ]
    ].copy()

    # One row per patient with diagnosis sequence
    patient_sequences = (
        df.groupby(["patient_no", "hyperedge_index", "sex_id", "sex", "age_group"])
        .agg(
            sequence=("node_id", list),
            icd_sequence=("icd_code", list),
            first_date_sequence=("first_date", list),
            n_unique_diagnoses=("node_id", "nunique")
        )
        .reset_index()
    )

    return hyperedge_table, patient_sequences

male_hyperedge_table, male_sequences = build_hypergraph_tables(male_diag)
female_hyperedge_table, female_sequences = build_hypergraph_tables(female_diag)

#male_hyperedge_table.head(5)
#male_sequences.head(5)


"""SAVING PART"""

# save hypergraph outputs

male_hyperedge_table.to_csv("male_hyperedge_table.csv", index=False)
female_hyperedge_table.to_csv("female_hyperedge_table.csv", index=False)
male_sequences.to_csv("male_sequences.csv", index=False)
female_sequences.to_csv("female_sequences.csv", index=False)
