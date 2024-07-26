import pandas as pd
import os

def extract_and_save_text(csv_file_path, output_directory):
    """
    Extract the 'article' and 'summary' columns from a CSV file and save them as .txt files.

    Parameters:
    csv_file_path (str): The path to the CSV file.
    output_directory (str): The directory where the .txt files will be saved.
    """
    # Create the output directory if it doesn't exist
    os.makedirs(output_directory, exist_ok=True)
    
    # Read the CSV file
    df = pd.read_csv(csv_file_path)
    
    # Ensure the CSV file contains the required columns
    if 'article' not in df.columns or 'highlights' not in df.columns:
        raise ValueError("CSV file must contain 'article' and 'summary' columns")
    
    # Extract and save the articles
    articles = df['article'].tolist()
    with open(os.path.join(output_directory, 'articles.txt'), 'w', encoding='utf-8') as article_file:
        for article in articles:
            article_file.write(article + '\n\n')  # Separate articles by two new lines
    
    # Extract and save the summaries
    summaries = df['highlights'].tolist()
    with open(os.path.join(output_directory, 'summaries.txt'), 'w', encoding='utf-8') as summary_file:
        for summary in summaries:
            summary_file.write(summary + '\n\n')  # Separate summaries by two new lines