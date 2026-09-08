from pathlib import Path
import re
from statistics import mean
from litellm import token_counter

LOGS_DIR = Path("logs")
MODEL_NAME = "gemini-3-flash-preview"


def extract_prompt_section(text: str) -> str:
    start_match = re.search(r"(?m)^FINAL BUILT PROMPT\s*$", text)
    if not start_match:
        return ""

    prompt_text = text[start_match.end():]

    end_match = re.search(r"(?m)^MODEL RESPONSE\s*$", prompt_text)
    if end_match:
        prompt_text = prompt_text[:end_match.start()]

    return prompt_text.strip()


def extract_messages(text: str) -> dict[int, str]:
    pattern = re.compile(
        r"(?ms)^\[MESSAGE\s+(\d+)\]\s*\n(.*?)(?=^\[MESSAGE\s+\d+\]\s*\n|\Z)"
    )

    messages = {}
    for match in pattern.finditer(text):
        msg_num = int(match.group(1))
        msg_text = match.group(2).strip()
        messages[msg_num] = msg_text

    return messages


def strip_role_and_content_prefix(msg_text: str) -> str:
    marker = "content:"
    pos = msg_text.find(marker)
    if pos == -1:
        return msg_text.strip()
    return msg_text[pos + len(marker):].strip()


def count_tokens(text: str, model: str) -> int:
    return token_counter(model=model, text=text)


def main():
    txt_files = sorted(
        p for p in LOGS_DIR.rglob("*.txt")
        if "run_evaluation" not in p.parts
    )

    if not txt_files:
        print(f"No .txt files found under {LOGS_DIR.resolve()}")
        return

    token_lengths = {i: [] for i in range(1, 6)}
    percentage_lengths = {i: [] for i in range(1, 6)}

    files_used = 0

    for path in txt_files:
        try:
            raw = path.read_text(encoding="utf-8", errors="replace")
            prompt_only = extract_prompt_section(raw)

            if not prompt_only:
                continue

            messages = extract_messages(prompt_only)

            if not all(i in messages for i in range(1, 6)):
                continue

            # count tokens for each message in this file
            file_token_counts = {}

            for i in range(1, 6):
                clean_text = strip_role_and_content_prefix(messages[i])
                n_tokens = count_tokens(clean_text, MODEL_NAME)

                token_lengths[i].append(n_tokens)
                file_token_counts[i] = n_tokens

            total_tokens = sum(file_token_counts.values())

            # compute percentages for this file
            for i in range(1, 6):
                pct = (file_token_counts[i] / total_tokens) * 100
                percentage_lengths[i].append(pct)

            files_used += 1

        except Exception as e:
            print(f"Error processing {path}: {e}")

    if files_used == 0:
        print("No valid files found.")
        return

    print(f"\nProcessed {files_used} files.\n")

    print("=== Average Token Lengths ===")
    for i in range(1, 6):
        avg_len = mean(token_lengths[i])
        print(f"MESSAGE {i}: {avg_len:.2f} tokens")

    print("\n=== Average Percentage of Prompt ===")
    for i in range(1, 6):
        avg_pct = mean(percentage_lengths[i])
        print(f"MESSAGE {i}: {avg_pct:.2f}%")



if __name__ == "__main__":
    main()