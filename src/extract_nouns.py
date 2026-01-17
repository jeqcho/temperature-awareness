#!/usr/bin/env python3
"""Extract nouns from HTML file and save as line-separated text file."""

import re
from pathlib import Path

def extract_nouns(html_path: str, output_path: str) -> int:
    """
    Extract nouns from the HTML file and save to a text file.
    
    Args:
        html_path: Path to the input HTML file
        output_path: Path to the output text file
        
    Returns:
        Number of nouns extracted
    """
    # Read the HTML file
    with open(html_path, 'r', encoding='utf-8') as f:
        html_content = f.read()
    
    # Find all nouns - they appear in <td width="120"> elements
    # Pattern 1: Nouns wrapped in <a> tags: <a href="/how-to-use/NOUN" target="_blank">NOUN</a>
    # Pattern 2: Plain text nouns in <td width="120">NOUN</td>
    
    # Extract nouns from anchor tags with how-to-use links
    anchor_pattern = r'<a href="/how-to-use/[^"]*" target="_blank">([^<]+)</a>'
    anchor_nouns = re.findall(anchor_pattern, html_content)
    
    # Extract nouns from plain td elements (for words like "two")
    # These are in <td width="120">\n\t\t\t\t\t\t\tWORD\n</td> pattern without anchor
    # We need to find td width="120" that don't contain anchor tags
    td_pattern = r'<td width="120">\s*\n\s*([a-zA-Z][a-zA-Z\-\']*)\s*\n\s*</td>'
    plain_nouns = re.findall(td_pattern, html_content)
    
    # Combine and clean nouns
    all_nouns = []
    
    # Process anchor nouns first (in order of appearance)
    for noun in anchor_nouns:
        noun = noun.strip()
        if noun and noun not in all_nouns:
            all_nouns.append(noun)
    
    # Add plain nouns that aren't already in the list
    for noun in plain_nouns:
        noun = noun.strip()
        if noun and noun not in all_nouns:
            all_nouns.append(noun)
    
    # Actually, we need to preserve order from the HTML. Let's use a different approach
    # Find all rows and extract nouns in order
    all_nouns = []
    
    # Find all <td width="120"> content
    td_content_pattern = r'<td width="120">\s*((?:<a[^>]*>([^<]+)</a>)|([a-zA-Z][a-zA-Z\-\']*))?\s*</td>'
    
    # Simpler approach: find the GridView3 table and extract systematically
    # Look for either: <a href="/how-to-use/WORD"...>WORD</a> or plain text
    
    # Pattern to match both cases
    combined_pattern = r'<td width="120">\s*(?:<a href="/how-to-use/[^"]*" target="_blank">([^<]+)</a>|([a-zA-Z][a-zA-Z\-\']*)\s*\n)'
    
    matches = re.findall(combined_pattern, html_content)
    
    for match in matches:
        # match is a tuple (anchor_noun, plain_noun)
        noun = match[0] if match[0] else match[1]
        if noun:
            noun = noun.strip()
            if noun and noun != 'Word':  # Skip the header "Word"
                all_nouns.append(noun)
    
    # Write to output file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(all_nouns) + '\n')
    
    return len(all_nouns)


if __name__ == '__main__':
    html_file = '/home/ubuntu/temperature-awareness/data/Top 1500 Nouns used in English Vocabulary Words for Speaking.html'
    output_file = '/home/ubuntu/temperature-awareness/data/nouns.txt'
    
    count = extract_nouns(html_file, output_file)
    print(f"Extracted {count} nouns to {output_file}")
