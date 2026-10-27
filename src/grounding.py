"""Deterministic check against the source: is every extracted string actually in the resume text?

Needs no golden labels, so it runs on all 60 documents of a run.
Checked: title, city, desired_salary, work_location_preference, additional_educations, skills,
work_experiences.position/company/city, educations.institution/faculty/city.
Not checked: enums, languages and dates - the prompt asks the model to translate / normalise them.
"Дистанційно" counts as grounded when the text mentions remote work in any wording (REMOTE).

A value is grounded if it occurs in the text as is, or if every word of it (3+ letters) occurs in the
text by its stem (the word without the last two letters). Stems absorb Ukrainian/Russian case endings
("у м.Вінниці" -> "Вінниця"); a paraphrase the model wrote itself does not pass.
"""
import re

TOP = ["title", "city", "work_location_preference", "additional_educations", "skills"]
EXP = ["position", "company", "city"]
EDU = ["institution", "faculty", "city"]
# "Дистанційно" is the normalised value the prompt asks for; the text says it in its own words
REMOTE = ["дистанц", "віддален", "віддалено", "удален", "удаленн", "remote", "онлайн", "online"]


def gnorm(s: str) -> str:
    s = s.lower().replace("’", "'").replace("ʼ", "'").replace("ё", "е")
    return " ".join(re.findall(r"[\w']+", s))


def values(out: dict) -> list[tuple[str, str]]:
    """(field, value) pairs to check in one extraction."""
    pairs = []
    for f in TOP:
        v = out.get(f)
        pairs += [(f, x) for x in (v if isinstance(v, list) else [v]) if x]
    if out.get("desired_salary"):
        pairs.append(("desired_salary", str(out["desired_salary"])))
    for e in out.get("work_experiences") or []:
        pairs += [(f"work_experiences.{k}", e[k]) for k in EXP if e.get(k)]
    for e in out.get("educations") or []:
        pairs += [(f"educations.{k}", e[k]) for k in EDU if e.get(k)]
    return pairs


def found(value: str, text: str, digits: str) -> bool:
    v = gnorm(value)
    if not v or v in text:
        return True
    if v == "дистанційно":
        return any(w in text for w in REMOTE)
    if v.isdigit():                                     # salary: "15 000 грн" in text, 15000 in output
        return v in digits
    words = [w for w in v.split() if len(w) >= 3]
    return bool(words) and all(w[:max(3, len(w) - 2)] in text for w in words)


def check_doc(out: dict, text: str) -> list[tuple[str, str]]:
    """Ungrounded (field, value) pairs of one extraction."""
    t = gnorm(text)
    digits = re.sub(r"(?<=\d)[\s.,](?=\d{3})", "", text)
    return [(f, v) for f, v in values(out) if not found(v, t, digits)]


def summarize_grounding(records: list[dict], texts: dict[str, str]) -> dict:
    total, bad, by_field, examples = 0, 0, {}, []
    for r in records:
        if not r["output"]:
            continue
        total += len(values(r["output"]))
        for f, v in check_doc(r["output"], texts[r["doc_id"]]):
            bad += 1
            key = f.split(".")[0]
            by_field[key] = by_field.get(key, 0) + 1
            examples.append({"doc_id": r["doc_id"], "field": f, "value": v})
    return {"values_checked": total, "ungrounded": bad,
            "ungrounded_pct": round(100 * bad / total, 1) if total else None,
            "ungrounded_by_field": dict(sorted(by_field.items(), key=lambda x: -x[1])),
            "ungrounded_examples": examples}
