import re
import nltk
from nltk.tokenize import sent_tokenize
import spacy
from tqdm import tqdm
import pandas as pd
import pickle
from nltk import pos_tag, word_tokenize

nlp = spacy.load('en_core_web_sm')


def get_subphrase_pos_tags(full_phrase, phrase):
    """
    Extract POS tags for a sub-phrase from the full phrase and its POS tags.

    Parameters:
        full_phrase (str): The full phrase as a string.
        full_pos_tags (list): List of POS tags for the full phrase, e.g., [('word', 'POS', 'dep', 'head'), ...].
        phrase_tokens (list): The tokens of the sub-phrase to extract POS tags for.

    Returns:
        list: POS tags corresponding to the tokens in the sub-phrase.
    """
    # Use SpaCy to tokenize the full phrase
    cleaned_full_phrase = " ".join(full_phrase.split())
    doc = nlp(cleaned_full_phrase)
    doc_tokens = [token.text for token in doc if not token.is_punct]
    
    cleaned_phrase_tokens = " ".join(phrase.split())
    phrase_tokens = [token.text for token in nlp(cleaned_phrase_tokens) if not token.is_punct]
    full_pos_tags = [(token.text, token.pos_, token.dep_, token.head.text) for token in doc if not token.is_punct]
    
    # Find the starting index of the sub-phrase in the full phrase
    match_found = False
    for start_idx in range(len(doc_tokens) - len(phrase_tokens) + 1):
        if doc_tokens[start_idx:start_idx + len(phrase_tokens)] == phrase_tokens:
            match_found = True
            break

    if not match_found:
        print(full_phrase)
        print(phrase)
        print("doc_tokens: ", doc_tokens)
        print("full_phrase: ", full_phrase)
        print("phrase_tokens: ", phrase_tokens)
        print("full_pos_tags: ", full_pos_tags)
        raise ValueError("Sub-phrase not found in full phrase.")

    # Extract the corresponding POS tags
    subphrase_pos_tags = full_pos_tags[start_idx:start_idx + len(phrase_tokens)]

    return subphrase_pos_tags


def extract_emotion_phrases(text):
    """
    Extract phrases based on specific patterns:
    1. "I" + up-to-3-words-in-between + feel/felt/am feeling/etc + up-to-4-word (including comma).
    2. "I" + "am/was/have been/had been" or "I'm/Im" + no-the-word-"feeling"-after + up-to-3-word (including comma).
    3. "feeling" with no noun, proper noun, or pronoun before it.

    Returns:
        phrases (list): Extracted short phrases.
        positions (list): Start positions of the extracted phrases.
        patterns (list): Flags indicating specific pattern rules.
        full_phrases (list): Original full patterns matched in the text.
        pos_tags (list): POS tags of the words in the extracted short phrases.
    """
    phrases = []
    positions = []
    patterns = []
    full_phrases = []  # To store the full matched patterns
    pos_tags = []
    
    # Pattern 1: "I" + up-to-3-words-in-between + feel/related words
    pattern1 = re.compile(r"\b[Ii]\b(?:\s+\w+){0,3}?\s+(?:feel|felt|am feeling|feeling|was feeling|had felt|have felt)\s+((?:\w+(?:\s*,?\s*\w+){0,3}))", re.IGNORECASE)

    # Pattern 2: "I" + "am/was/have been/had been" or "I'm/Im"
    pattern2 = re.compile(r"\b(?:[Ii](?:'m|m| am| was| have been| had been))\b\s+((?:\b(?!feeling)\w+\b(?:\s*,?\s*\b(?!feeling)\w+\b){0,2}))", re.IGNORECASE)
        
    # Process text for patterns 1 and 2
    for match in pattern1.finditer(text):
        full_phrase = match.group(0)
        phrase = match.group(1)  # Extracted phrase
        
        full_phrases.append(full_phrase)  # Full pattern match
        phrases.append(phrase)  # Extracted phrase
        positions.append(match.start(1))  # Start position of the extracted phrase
        patterns.append(False)
        
        phrase_pos_tags = get_subphrase_pos_tags(full_phrase, phrase)
        pos_tags.append(phrase_pos_tags)
        
    for match in pattern2.finditer(text):
        full_phrase = text[match.start():match.end()]
        phrase = match.group(1)  # Extracted phrase
        
        full_phrases.append(full_phrase)  # Full pattern match
        phrases.append(phrase)  # Extracted phrase
        positions.append(match.start(1))  # Start position of the extracted phrase
        patterns.append(True)

        # Use SpaCy POS tagging
        phrase_pos_tags = get_subphrase_pos_tags(full_phrase, phrase)
        pos_tags.append(phrase_pos_tags)
    
    # Pattern 3: Process for standalone "feeling" in all sentences
    doc = nlp(text)
    for sent in doc.sents:
        if "feeling" in sent.text.lower():
            tokens = [token for token in sent]
            for i, token in enumerate(tokens):
                if token.text.lower() == "feeling":
                    # Ensure no noun, proper noun, or pronoun before "feeling"
                    if all(
                        t.pos_ not in {"NOUN", "PROPN", "PRON"}
                        for t in tokens[:i]
                    ):
                        # Extract up to four non-punctuation words following "feeling"
                        following_words = []
                        for t in tokens[i + 1:]:
                            if not t.is_punct:
                                following_words.append(t)
                            if len(following_words) >= 4:
                                break
    
                        # Construct the phrase including original punctuation
                        phrase_tokens = tokens[i + 1 : i + 1 + len(following_words)]
                        phrase = " ".join(t.text for t in phrase_tokens).strip()
    
                        if phrase:
                            # Include up to three tokens before "feeling" and the tokens in `phrase_tokens`
                            full_phrase_tokens = tokens[max(0, i - 3) : i + 1 + len(phrase_tokens)]
                            full_phrase = " ".join(t.text for t in full_phrase_tokens).strip()

                            phrase_pos_tags = get_subphrase_pos_tags(full_phrase, phrase)
        
                            phrases.append(phrase)
                            positions.append(sent.start_char + token.idx + len(token) + 1)
                            patterns.append(False)
                            full_phrases.append(full_phrase)  # Store the full phrase
                            pos_tags.append(phrase_pos_tags)
                            break  # Only process the first match in the sentence

    return phrases, positions, patterns, full_phrases, pos_tags


def mask_emotion_words(phrases, emotion_list, emotion_list_adj, positions, patterns, pos_tags_list):
    """
    Masks emotion words in phrases based on specific rules:
    1. Avoids masking the word immediately following [MASK] + prepositions.
    2. Avoids the phrase "I feel like" or "I am like".
    3. Avoids masking if there is a pronoun, noun, or verb before the emotional word.
    4. Avoids masking if there is when/where/what/how or similar words before the emotional word.
    6. If the pattern is "I am", the emotional word must also be an adjective.
    7. If there is a word matching `pronouns_pattern` before the emotional word, do not mask it.

    Parameters:
        phrases (list): List of phrases to process.
        emotion_list (list): List of emotion words to be masked.
        positions (list): List of starting positions for phrases.
        patterns (list): List of booleans indicating whether the emotional word must be an adjective.
        pos_tags_list (list): List of POS tags (from SpaCy) for the phrases.

    Returns:
        list: List of masked positions.
    """
    pronouns_pattern = r"\b(?:she|he|we|they|this|that|these|those|i|you|it|me|us|him|her|them|my|your|his|her|our|their|mine|yours|hers|ours|theirs|she's|he's|we're|they're|you're|I'm|being|others|other)\b"
    prepositions = ["for", "at", "of", "by", "in", "on", "to", "with", "about", "as", "into", "onto"]
    question_words = {"when", "where", "how", "what", "why", "which"}
    masked_phrases = []
    masked_emotions = []

    for i, phrase in enumerate(phrases):
    
        must_be_adjective = patterns[i]  # If True, emotional word must be an adjective
        pos_tags = pos_tags_list[i]
        modified_phrase = []
        masked_emotion = []

        # If the phrase starts with "I feel like" or "I am like", skip it
        words = phrase.split()
        if words[0].lower() == 'like':
            masked_phrases.append(phrase)
            continue

        for j, (word, tag, dep, head) in enumerate(pos_tags):
            # Clean the word for comparison
            base_word = word.strip('.,!?;:"()[]').lower()

            # Check if the cleaned word is in emotion_list
            if base_word in emotion_list:
                # Check for disqualifying conditions
                # 1. Check for a pronoun before the emotional word
                if any(re.match(pronouns_pattern, w, flags=re.IGNORECASE) for w, _, _, _ in pos_tags[:j]):
                    modified_phrase.append(word)  # Do not mask

                # 2. Check for a verb before the emotional word
                elif any(tag in {"VERB", "PROPN", "AUX"} for _, tag, _, _ in pos_tags[:j]):
                    modified_phrase.append(word)  # Do not mask
                
                # 3. Check for a noun before, except when the emotional word is an "acomp" of "feel"
                elif any(tag == "NOUN" for _, tag, _, _ in pos_tags[:j]) and (dep == "acomp" and head not in {"feel", "felt", "feeling"}):
                    modified_phrase.append(word)  # Do not mask
                    
                # 4. Check for question words like when/where/how/what
                elif any(base_word in question_words for base_word, _, _, _ in pos_tags[:j]):
                    modified_phrase.append(word)  # Do not mask
                    
                # 5. Check for [MASK] + preposition before the word
                elif j > 1 and pos_tags[j - 1][0].lower() in prepositions and pos_tags[j - 2][0] == "[MASK]":
                    modified_phrase.append(word)

                # 6. If patterns[i] == True, check that the word is an adjective
                elif must_be_adjective and (tag not in {"ADJ"} or base_word not in emotion_list_adj):
                    modified_phrase.append(word)  # Do not mask if not an adjective

                else:
                    # Mask the emotion word
                    masked_emotion.append(base_word)  # Append cleaned word to masked_emotion
                    modified_phrase.append(f"[MASK]{word[len(base_word):]}")  # Preserve punctuation

            else:
                modified_phrase.append(word)

        masked_phrases.append(" ".join(modified_phrase))

    # Calculate masked positions
    masked_positions = []
    for i, masked_phrase in enumerate(masked_phrases):
        masked_position = positions[i] + masked_phrase.find('[MASK]')
        if masked_phrase.find('[MASK]') != -1:
            masked_positions.append(masked_position)

    masked_positions = list(set(masked_positions))

    return masked_positions


def replace_words_at_positions(text, positions):
    """
    Replaces words starting at specific positions with [MASK] and stores the replaced words in a list.
    
    Parameters:
        text (str): The original input text.
        positions (list): The list of starting positions of the words to be replaced.

    Returns:
        tuple: (new_text, replaced_words)
               - new_text: The text with the words replaced by [MASK].
               - replaced_words: The list of words that were replaced.
    """
    # Sort positions to handle them in order
    positions = sorted(positions)
    new_text = text
    replaced_words = []
    offset = 0

    for position in positions:
        adjusted_position = position + offset

        # Find the word and punctuation starting at the given position
        match = re.match(r"(\w+(['-]\w+)?)([.,;!?]*)", new_text[adjusted_position:])
        
        if match:
            # Extract the word and any following punctuation
            word = match.group(1)
            punctuation = match.group(3)

            # Record the replaced word
            replaced_words.append(word)

            # Replace the word with [MASK], keeping any punctuation
            start_position = adjusted_position
            end_position = start_position + len(word) + len(punctuation)

            # Construct new text by replacing the target word with [MASK]
            new_text = new_text[:start_position] + '[MASK]' + punctuation + new_text[end_position:]

            # Update offset to keep track of changes in text length
            offset += len('[MASK]') - len(word)

    return new_text, replaced_words

def mask_emotions(text, emotion_list, emotion_list_adj):
    phrases, positions, patterns, full_phrases, pos_tags = extract_emotion_phrases(text)
    masked_positions = mask_emotion_words(phrases, emotion_list, emotion_list_adj, positions, patterns, pos_tags)
    aug_text, emotions = replace_words_at_positions(text, masked_positions)
    return aug_text, emotions

def process_large_dataframe_in_chunks(input_file, output_file, emotion_list, emotion_list_adj, chunk_size=100):
    """
    Processes a large DataFrame in chunks, applies replace_phrases_with_masks() on the 'text' column,
    and appends the results to a new output file.

    Parameters:
        input_file (str): Path to the input CSV file containing the 'text' column.
        output_file (str): Path to the output CSV file to store results.
        emotion_list (list): List of emotion words for replace_phrases_with_masks().
        chunk_size (int): Number of rows to process at a time.
    """
    # Create an empty output file with the correct structure
    pd.DataFrame(columns=['index', 'id', 'created_utc', 'subreddit', 'title', 'selftext',
                          'text', 'word_count', 'sentence_count', 'aug', 'emotions']).to_csv(output_file, index=False)

    # Read and process the input file in chunks
    for chunk in tqdm(pd.read_csv(input_file, chunksize=chunk_size), desc="Processing chunks"):
        # Ensure 'text' column exists in the chunk
        if 'text' not in chunk.columns:
            raise ValueError("Input file must contain a 'text' column.")

        # Apply the function to the 'text' column
        processed_data = chunk['text'].apply(
            lambda x: mask_emotions(x, emotion_list, emotion_list_adj)
        )

        # Split the tuple output of replace_phrases_with_masks into separate columns
        chunk[['aug', 'emotions']] = pd.DataFrame(processed_data.tolist(), index=chunk.index)

        # Append the processed chunk to the output file
        chunk.to_csv(output_file, mode='a', index=False, header=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Emotion masking script for large CSV files.')
    parser.add_argument('--input', required=True, help='Path to input CSV file.')
    parser.add_argument('--output', required=True, help='Path to output CSV file.')
    parser.add_argument('--emotion_list', default='list_of_emotions.pkl', help='Path to emotion list pickle file.')
    parser.add_argument('--emotion_list_adj', default='list_of_emotions_adj.pkl', help='Path to emotion adjective list pickle file.')
    parser.add_argument('--chunk_size', type=int, default=100, help='Chunk size for processing.')

    args = parser.parse_args()

    with open(args.emotion_list, "rb") as file:
        emotion_list = pickle.load(file)

    with open(args.emotion_list_adj, "rb") as file:
        emotion_list_adj = pickle.load(file)

    process_large_dataframe_in_chunks(args.input, args.output, emotion_list, emotion_list_adj, args.chunk_size)