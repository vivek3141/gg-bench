import re


def extract_python_code(markdown_content: str) -> str | None:
    """
    Extracts Python code blocks from markdown content.

    Args:
        markdown_content (str): The markdown text to extract code from.

    Returns:
        str | None: Extracted Python code as a string, or None if no code found.
    """
    pattern = r"```python\s*(.*?)\s*```"
    matches = re.findall(pattern, markdown_content, re.DOTALL)
    if not matches:
        return None
    extracted_code = "\n\n".join(matches)
    return extracted_code
