def structure_output(predictions):
    """
    Given list of predictions (each with text, label, page), return structured JSON
    with document title and an outline of headings (H1, H2, H3).
    """
    result = {
        "title": "",
        "outline": []
    }

    title_found = False

    for item in predictions:
        label = item["label"]
        text = item["text"]
        page = item.get("page", -1)

        if label == "TITLE" and not title_found:
            result["title"] = text
            title_found = True
        elif label in {"H1", "H2", "H3"}:
            result["outline"].append({
                "level": label,
                "text": text,
                "page": page
            })

    return result
