"""Versioned T5 prompt semantics; cache keys remain original task text."""

RAW_TASK = "raw_task_v1"
WSA_INSTRUCTION = "wsa_instruction_v1"
PROMPT_FORMATS = (RAW_TASK, WSA_INSTRUCTION)
WSA_PROMPT = "A video recorded from a robot's point of view executing the following instruction: {task}"


def format_wan_prompt(task: str, prompt_format: str) -> str:
    if prompt_format not in PROMPT_FORMATS:
        raise ValueError(f"unsupported Wan prompt format: {prompt_format!r}")
    text = str(task).strip()
    if not text:
        raise ValueError("Wan task text must be non-empty")
    return WSA_PROMPT.format(task=text) if prompt_format == WSA_INSTRUCTION else text
