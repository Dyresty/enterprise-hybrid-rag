"""Ingest EU Horizon Europe projects from CORDIS into the RAG index.

Download first (about 40 MB):
    https://cordis.europa.eu/data/cordis-HORIZONprojects-csv.zip  -> unzip into data/cordis/

Corpus: every project with Universitaet Paderborn as participant, plus computer-science
projects with a German participant (capped by --limit). One CORDIS project = one document.

    python -m scripts.ingest_cordis --limit 1500
"""
import argparse
import json
from pathlib import Path

import pandas as pd

from app.services.container import get_container

DATA = Path("data/cordis")


def read(name):
    return pd.read_csv(DATA / name, sep=";", dtype=str, encoding="utf-8", on_bad_lines="skip").fillna("")


def select_projects(projects, orgs, fields, limit):
    paderborn = set(orgs.loc[orgs["name"].str.contains("PADERBORN", case=False), "projectID"])
    german = set(orgs.loc[orgs["country"] == "DE", "projectID"])
    computing = set(fields.loc[fields["euroSciVocPath"].str.contains("computer and information sciences"), "projectID"])
    others = sorted((german & computing) - paderborn)[: max(0, limit - len(paderborn))]
    return projects[projects["id"].isin(paderborn | set(others))]


def project_text(p, participants, fields):
    coordinator = participants[participants["role"] == "coordinator"]
    partners = participants[participants["role"] != "coordinator"]
    return "\n".join(
        [
            f"Project: {p['acronym']} - {p['title']}",
            f"Grant ID: {p['id']} | Programme: {p['frameworkProgramme']} | Funding scheme: {p['fundingScheme']}",
            f"Call: {p['masterCall']} | Topic: {p['topics']}",
            f"Duration: {p['startDate']} to {p['endDate']} | Status: {p['status']}",
            f"Total cost: {p['totalCost']} EUR | EU contribution: {p['ecMaxContribution']} EUR",
            f"Coordinator: {', '.join(coordinator['name'] + ' (' + coordinator['country'] + ')')}",
            f"Participants: {', '.join((partners['name'] + ' (' + partners['country'] + ')').head(25))}",
            f"Scientific fields: {', '.join(sorted(set(fields)))}",
            f"Keywords: {p['keywords']}",
            "",
            f"Objective: {p['objective']}",
        ]
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=1500)
    args = parser.parse_args()

    projects, orgs, fields = read("project.csv"), read("organization.csv"), read("euroSciVoc.csv")
    selected = select_projects(projects, orgs, fields, args.limit)
    orgs_by_project = dict(tuple(orgs.groupby("projectID")))
    fields_by_project = fields.groupby("projectID")["euroSciVocTitle"].apply(list).to_dict()
    print(f"Selected {len(selected)} projects")

    ingestion = get_container().ingestion
    corpus = []
    for i, (_, p) in enumerate(selected.iterrows(), start=1):
        text = project_text(p, orgs_by_project.get(p["id"], orgs.iloc[0:0]), fields_by_project.get(p["id"], []))
        title = f"{p['acronym']} - {p['title']}"
        url = f"https://cordis.europa.eu/project/id/{p['id']}"
        ingestion.ingest_pages(
            [{"page": 1, "text": text}], p["id"], f"CORDIS {p['id']}", "cordis", title, url, refresh=False
        )
        corpus.append({"id": p["id"], "title": title, "text": text})
        if i % 200 == 0:
            print(f"  {i}/{len(selected)}")

    ingestion.refresh_bm25()
    # Saved so the evaluation set can be written against exactly this corpus.
    with open(DATA / "corpus.jsonl", "w", encoding="utf-8") as f:
        for row in corpus:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Done. Chunks in index: {get_container().vector_store.count()}")
    get_container().vector_store.close()  # embedded Qdrant errors if left to close at interpreter exit


if __name__ == "__main__":
    main()
