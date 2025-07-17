# app/output_format.py

def structure_output(predictions):
    """
    Given a list of (text, label) tuples, return structured JSON.
    """

    output = {
        "TITLE": None,
        "sections": []
    }

    current_section = None

    for text, label in predictions:
        if label == "TITLE":
            output["TITLE"] = text

        elif label in ("H1", "H2", "H3"):
            # Start a new section
            current_section = {
                "heading": text,
                "content": []
            }
            output["sections"].append(current_section)

        elif label == "BODY":
            if current_section:
                current_section["content"].append(text)
            else:
                # BODY without heading — put it in a default section
                if not output["sections"]:
                    output["sections"].append({
                        "heading": "Untitled",
                        "content": [text]
                    })
                    current_section = output["sections"][-1]
                else:
                    output["sections"][-1]["content"].append(text)

    return output
