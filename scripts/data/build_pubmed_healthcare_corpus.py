#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
import xml.etree.ElementTree as ET
from datetime import date
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen


REPO_ROOT = Path(__file__).resolve().parents[2]
PUBMED_ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
PUBMED_EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
DEFAULT_QUERY = (
    '("healthcare"[Title/Abstract] OR "health care"[Title/Abstract] OR '
    '"hospital"[Title/Abstract] OR "patient care"[Title/Abstract] OR '
    '"clinical"[Title/Abstract]) AND english[lang] AND hasabstract[text] '
    'AND pubmed pmc open access[filter]'
)
DEFAULT_FROM_DATE = "2025/01/01"
DEFAULT_FETCH_MAX = 2200
DEFAULT_SPLIT_COUNT = 500
DEFAULT_BATCH_SIZE = 100
REQUEST_DELAY_SECONDS = 0.4
MIN_TEXT_CHARS = 600
MAX_TEXT_CHARS = 4000


def normalize_whitespace(text: str) -> str:
    return " ".join(text.split()).strip()


def clip_text(text: str, max_chars: int = MAX_TEXT_CHARS) -> str:
    text = normalize_whitespace(text)
    if len(text) <= max_chars:
        return text
    clipped = text[:max_chars]
    last_period = clipped.rfind(". ")
    if last_period > 200:
        clipped = clipped[: last_period + 1]
    return clipped.strip()


def fetch_esearch_ids(*, query: str, from_date: str, to_date: str, retmax: int) -> list[str]:
    params = {
        "db": "pubmed",
        "term": query,
        "sort": "pub date",
        "retmode": "json",
        "retmax": str(retmax),
        "mindate": from_date,
        "maxdate": to_date,
        "datetype": "pdat",
    }
    with urlopen(f"{PUBMED_ESEARCH_URL}?{urlencode(params)}", timeout=60) as response:
        payload = json.loads(response.read().decode("utf-8"))
    return [str(item) for item in payload.get("esearchresult", {}).get("idlist", [])]


def parse_pubmed_articles(xml_payload: str) -> list[dict[str, str]]:
    root = ET.fromstring(xml_payload)
    articles: list[dict[str, str]] = []

    for article in root.findall(".//PubmedArticle"):
        pmid = normalize_whitespace(article.findtext(".//MedlineCitation/PMID", default=""))
        title = normalize_whitespace("".join(article.find(".//ArticleTitle").itertext())) if article.find(".//ArticleTitle") is not None else ""
        abstract_nodes = article.findall(".//Abstract/AbstractText")
        abstract_parts = [normalize_whitespace("".join(node.itertext())) for node in abstract_nodes]
        abstract_parts = [part for part in abstract_parts if part]
        abstract = " ".join(abstract_parts).strip()
        journal = normalize_whitespace(article.findtext(".//Journal/Title", default=""))
        pub_year = normalize_whitespace(article.findtext(".//PubDate/Year", default=""))
        month = normalize_whitespace(article.findtext(".//PubDate/Month", default=""))
        day = normalize_whitespace(article.findtext(".//PubDate/Day", default=""))

        if not pmid or not title or not abstract:
            continue

        pmcid = ""
        for article_id in article.findall(".//PubmedData/ArticleIdList/ArticleId"):
            if article_id.attrib.get("IdType") == "pmc":
                pmcid = normalize_whitespace("".join(article_id.itertext()))
                break

        articles.append(
            {
                "pmid": pmid,
                "pmcid": pmcid,
                "title": title,
                "abstract": abstract,
                "journal": journal,
                "publication_date": "-".join(part for part in [pub_year, month, day] if part),
            }
        )
    return articles


def fetch_pubmed_articles(pmids: list[str], batch_size: int) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    for start in range(0, len(pmids), batch_size):
        batch = pmids[start : start + batch_size]
        params = {
            "db": "pubmed",
            "id": ",".join(batch),
            "retmode": "xml",
        }
        with urlopen(f"{PUBMED_EFETCH_URL}?{urlencode(params)}", timeout=90) as response:
            xml_payload = response.read().decode("utf-8")
        records.extend(parse_pubmed_articles(xml_payload))
        time.sleep(REQUEST_DELAY_SECONDS)
    return records


def to_cpt_record(article: dict[str, str], split: str, index: int) -> dict[str, str]:
    text = clip_text(f"{article['title']}. {article['abstract']}")
    return {
        "id": f"{split}-{index + 1:04d}",
        "text": text,
        "source": "pubmed-pmc-open-access",
        "language": "en",
        "domain": "healthcare",
        "split": split,
        "pmid": article["pmid"],
        "pmcid": article["pmcid"],
        "title": article["title"],
        "journal": article["journal"],
        "publication_date": article["publication_date"],
    }


def write_jsonl(path: Path, records: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(record, ensure_ascii=True) for record in records) + "\n", encoding="utf-8")


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def resolve_split_sizes(
    *,
    split_count: int | None,
    train_count: int | None,
    eval_count: int | None,
    test_count: int | None,
) -> tuple[int, int, int]:
    if split_count is not None:
        resolved_train = train_count if train_count is not None else split_count
        resolved_eval = eval_count if eval_count is not None else split_count
        resolved_test = test_count if test_count is not None else split_count
        return resolved_train, resolved_eval, resolved_test
    if train_count is None or eval_count is None or test_count is None:
        raise ValueError("Provide either --split-count or all of --train-count, --eval-count, and --test-count.")
    return train_count, eval_count, test_count


def build_dataset(
    *,
    query: str,
    from_date: str,
    to_date: str,
    train_count: int,
    eval_count: int,
    test_count: int,
    retmax: int,
    batch_size: int,
    output_dir: Path,
    metadata_path: Path,
) -> None:
    total_count = train_count + eval_count + test_count
    pmids = fetch_esearch_ids(query=query, from_date=from_date, to_date=to_date, retmax=retmax)
    if len(pmids) < total_count:
        raise RuntimeError(
            f"PubMed search returned only {len(pmids)} records; need at least {total_count}. "
            "Broaden the query or widen the date range."
        )

    articles = fetch_pubmed_articles(pmids, batch_size=batch_size)
    seen_pmids: set[str] = set()
    filtered_articles: list[dict[str, str]] = []
    for article in articles:
        if article["pmid"] in seen_pmids:
            continue
        seen_pmids.add(article["pmid"])
        text = clip_text(f"{article['title']}. {article['abstract']}")
        if len(text) < MIN_TEXT_CHARS:
            continue
        filtered_articles.append(article)
        if len(filtered_articles) >= total_count:
            break

    if len(filtered_articles) < total_count:
        raise RuntimeError(
            f"Only collected {len(filtered_articles)} usable abstracts; need {total_count}. "
            "Broaden the query or widen the date range."
        )

    # PubMed search results are newest-first. Keep the most recent records for
    # test, the next block for eval, and the oldest records in the selected
    # window for train to reduce leakage from future-looking sampling.
    selected = filtered_articles[:total_count]
    test_articles = selected[:test_count]
    eval_articles = selected[test_count : test_count + eval_count]
    train_articles = selected[test_count + eval_count : test_count + eval_count + train_count]

    train_records = [to_cpt_record(article, "train", idx) for idx, article in enumerate(train_articles)]
    eval_records = [to_cpt_record(article, "eval", idx) for idx, article in enumerate(eval_articles)]
    test_records = [to_cpt_record(article, "test", idx) for idx, article in enumerate(test_articles)]

    write_jsonl(output_dir / "train.jsonl", train_records)
    write_jsonl(output_dir / "eval.jsonl", eval_records)
    write_jsonl(output_dir / "test.jsonl", test_records)

    dataset_root = metadata_path.resolve().parent
    train_rel = output_dir.resolve().joinpath("train.jsonl").relative_to(dataset_root)
    eval_rel = output_dir.resolve().joinpath("eval.jsonl").relative_to(dataset_root)
    test_rel = output_dir.resolve().joinpath("test.jsonl").relative_to(dataset_root)
    metadata = {
        "dataset_name": f"qwen2-5-0-5b-healthcare-pubmed-cpt-{total_count}",
        "dataset_version": (
            f"pubmed-healthcare-{from_date.replace('/', '')}-{to_date.replace('/', '')}-"
            f"{train_count}-{eval_count}-{test_count}-v1"
        ),
        "format": "jsonl",
        "text_field": "text",
        "source": "pubmed-pmc-open-access",
        "language": "en",
        "query": query,
        "from_date": from_date,
        "to_date": to_date,
        "splits": {
            "train": str(train_rel),
            "eval": str(eval_rel),
            "test": str(test_rel),
        },
    }
    write_json(metadata_path, metadata)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a healthcare CPT corpus from recent PubMed/PMC Open Access abstracts.")
    parser.add_argument("--query", default=DEFAULT_QUERY)
    parser.add_argument("--from-date", default=DEFAULT_FROM_DATE)
    parser.add_argument("--to-date", default=date.today().strftime("%Y/%m/%d"))
    parser.add_argument("--split-count", type=int, default=DEFAULT_SPLIT_COUNT)
    parser.add_argument("--train-count", type=int)
    parser.add_argument("--eval-count", type=int)
    parser.add_argument("--test-count", type=int)
    parser.add_argument("--retmax", type=int, default=DEFAULT_FETCH_MAX)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument(
        "--output-dir",
        default=str(REPO_ROOT / "dataset" / "cpt" / "qwen2_5_0_5b_healthcare_pubmed_500"),
    )
    parser.add_argument(
        "--metadata-path",
        default=str(REPO_ROOT / "dataset" / "metadata.qwen2_5_0_5b.healthcare.pubmed.500.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_count, eval_count, test_count = resolve_split_sizes(
        split_count=args.split_count,
        train_count=args.train_count,
        eval_count=args.eval_count,
        test_count=args.test_count,
    )
    build_dataset(
        query=args.query,
        from_date=args.from_date,
        to_date=args.to_date,
        train_count=train_count,
        eval_count=eval_count,
        test_count=test_count,
        retmax=args.retmax,
        batch_size=args.batch_size,
        output_dir=Path(args.output_dir).resolve(),
        metadata_path=Path(args.metadata_path).resolve(),
    )
    print(Path(args.output_dir).resolve())
    print(Path(args.metadata_path).resolve())


if __name__ == "__main__":
    main()
