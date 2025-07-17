# app/output_format.py

def structure_output(predictions):
    """
    Given list of predictions (each with text, label, page), return structured JSON grouped by labels.
    """

    output = {
        "TITLE": [],
        "H1": [],
        "H2": [],
        "H3": [],
        "BODY": []
    }

    for item in predictions:
        label = item["label"]
        text = item["text"]

        if label in output:
            output[label].append(text)
        else:
            # Optionally collect unknown labels if needed
            print(f"⚠️ Unknown label: {label}")
    
    return output
