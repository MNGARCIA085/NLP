import re
import nltk

# Download necessary NLTK data
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Initialize lemmatizer and stop words
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    """
    Preprocess the input text by removing specific characters, hyperlinks, and stop words,
    and by tokenizing and lemmatizing the remaining words.
    
    Args:
    text (str): The text to be processed.
    
    Returns:
    str: The processed text.
    """
    try:
        # Remove specific characters (@ and #) and hyperlinks
        text = re.sub(r'[@#]', '', text)
        text = re.sub(r'https?:\/\/\S+', '', text)
        
        # Tokenize and lemmatize
        words = word_tokenize(text)
        filtered_words = [lemmatizer.lemmatize(word.lower()) for word in words if word.lower() not in stop_words]
        
        return ' '.join(filtered_words)
    except Exception as e:
        print(f"Error processing text: {e}")
        return ""



def store_results(results, name, score):
    """
    Add a result to the given dictionary.
    
    Args:
        results (dict): Dictionary to store the results.
        name (str): Name of the model.
        score(list): 
            - pos. 0 : loss
            - pos. 1 : accuracy, Accuracy of the model.
            - pos. 2 : precison, Precision of the model. 
            - pos. 3 : recall, Recall of the model.
    
    Returns:
        dict: The updated dictionary with the appended result.
    """
    # Calculate the F1-score
    f1_score = (2 * (score[2] * score[3])) / (score[2] + score[3]) if score[2] + score[3] != 0 else 0.0
    
    # Create the data dictionary
    data = {
        'accuracy': score[1],
        'precision': score[2],
        'recall': score[3],
        'f1_score': f1_score
    }
    
    # Store the data in the results dictionary
    results[name] = data
    
    return results



