import csv


def load_awesome_chatgpt_prompts(path: str):
    """
    Load prompts from a CSV file and return the prompt column as a list,
    along with a list containing the first half of each prompt text (split by words).
    
    Args:
        path: Path to the CSV file containing prompts
        
    Returns:
        Tuple of (full_prompts_list, half_prompts_list)
    """
    prompts = []
    
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            prompts.append(row['prompt'])
    
    half_prompts = []
    for prompt in prompts:
        words = prompt.split()
        half_length = len(words) // 2
        half_prompt = ' '.join(words[:half_length])
        half_prompts.append(half_prompt)
    
    return prompts, half_prompts