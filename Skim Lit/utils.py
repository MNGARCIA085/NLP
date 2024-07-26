import matplotlib.pyplot as plt


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



def plot_hist(data, title, xlabel, ylabel, bins_number = None):
    """
    Plots a histogram for the given data with optional bin number customization.

    Parameters:
        data: (pd.Series or np.ndarray): The data to plot.
        bins_number: (int or sequence, optional): Number of bins or the bin edges. If None, bins 
                        will be automatically determined.
    """
    
    # number of bins
    n_bins = bins_number if bins_number else range(1, data.max() + 2)
    
    # Plotting the histogram
    plt.figure(figsize=(10, 6))
    n, bins, patches = plt.hist(data, bins=n_bins, edgecolor='black')

    # Adding colors to the bins
    for patch, color in zip(patches, plt.cm.viridis(n / max(n))):
        patch.set_facecolor(color)

    # Adding titles and labels
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    # Show the plot
    plt.show()






