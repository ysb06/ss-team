import csv


def load_awesome_chatgpt_prompts(path: str):
    """
    Load prompts from a CSV file and return only the prompt column as a list.
    
    Args:
        path: Path to the CSV file containing prompts
        
    Returns:
        List of prompt strings
    """
    prompts = []
    
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            prompts.append(row['prompt'])
    
    return prompts