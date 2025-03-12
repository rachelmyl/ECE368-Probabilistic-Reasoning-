import os.path
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
import util
import random

def learn_distributions(file_lists_by_category):
    """
   Estimate the parameters p_d, and q_d from the training set
   
   Input
   -----
   file_lists_by_category: A two-element list. The first element is a list of 
   spam files, and the second element is a list of ham files.

   Output
   ------
   probabilities_by_category: A two-element tuple. The first element is a dict 
   whose keys are words, and whose values are the smoothed estimates of p_d;
   the second element is a dict whose keys are words, and whose values are the 
   smoothed estimates of q_d 
   """
   ### TODO: Write your code here
    spam_files, ham_files = file_lists_by_category
    vocabulary = set()
    spam_counts, ham_counts = Counter(), Counter()
    
    for files, counts in zip((spam_files, ham_files), (spam_counts, ham_counts)):
        for file in files:
            words = util.get_words_in_file(file)
            counts.update(words)
            vocabulary.update(words)
    
    D = len(vocabulary)
    total_spam, total_ham = sum(spam_counts.values()), sum(ham_counts.values())
    
    p_d = {word: (spam_counts[word] + 1) / (total_spam + D) for word in vocabulary}
    q_d = {word: (ham_counts[word] + 1) / (total_ham + D) for word in vocabulary}
    probabilities_by_category = p_d, q_d
    
    return probabilities_by_category

def classify_new_email(filename, probabilities_by_category, prior_by_category):
    """
    Use Naive Bayes classification to classify the email in the given file.

    Inputs
    ------
    filename: name of the file to be classified
    probabilities_by_category: output of function learn_distributions
    prior_by_category: A two-element list as [\pi, 1-\pi], where \pi is the 
    parameter in the prior class distribution

    Output
    ------
    classify_result: A two-element tuple. The first element is a string whose value
    is either 'spam' or 'ham' depending on the classification result, and the 
    second element is a two-element list as [log p(y=1|x), log p(y=0|x)], 
    representing the log posterior probabilities
    """
    ### TODO: Write your code here
    p_d, q_d = probabilities_by_category
    words = util.get_words_in_file(filename)
    word_counts = Counter(words)
    
    log_spam, log_ham = np.log10(prior_by_category[0]), np.log10(prior_by_category[1])
    
    for word, count in word_counts.items():
        if word in p_d:
            log_spam += count * np.log10(p_d[word])
            log_ham += count * np.log10(q_d[word])
    
    classify_result = ("spam", [log_spam, log_ham]) if log_spam > log_ham else ("ham", [log_spam, log_ham])
    
    return classify_result

def classify_new_email_mod(filename, probabilities_by_category, epsilon):
    pi = 1 / (epsilon + 1)
    return classify_new_email(filename, probabilities_by_category, (pi, 1 - pi))

def generate_tradeoff_plot(test_folder, probabilities_by_category):
    """
    Generate a Type 1 vs. Type 2 error tradeoff curve.
    """
    ratios = [10 ** exp for exp in np.linspace(-10, 10, 20)]
    type1_errors, type2_errors = [], []
    
    for ratio in ratios:
        performance_measures = np.zeros((2, 2))
        for filename in util.get_files_in_folder(test_folder):
            label, _ = classify_new_email_mod(filename, probabilities_by_category, ratio)
            true_label = 'ham' in os.path.basename(filename)
            guessed_label = (label == 'ham')
            performance_measures[int(true_label), int(guessed_label)] += 1
        
        correct = np.diag(performance_measures)
        totals = np.sum(performance_measures, axis=1)
        type1_errors.append(totals[0] - correct[0])
        type2_errors.append(totals[1] - correct[1])
    
    plt.plot(type1_errors, type2_errors)
    plt.scatter(type1_errors, type2_errors)
    plt.xlabel("Number of Type 1 errors")
    plt.ylabel("Number of Type 2 errors")
    plt.title("Trade-off between Type 1 and Type 2 errors")
    plt.savefig("nbc.pdf")
    plt.show()
    
def select_files(directory, fraction=0.7):
   """
   This function builds a customized dataset for each group
   """
   all_files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.txt')]
   random.shuffle(all_files)
   num_files = int(len(all_files) * fraction)
   return all_files[:num_files]
    
if __name__ == '__main__':
    ############################CHANGE YOUR STUDENT ID###############################
    student1_number = 1008864837  # Replace with the actual student number
    student2_number = 1008761751  # Replace with the actual student number
    random.seed((student1_number+student2_number)/1000)
      
    
    # folder for training and testing 
    spam_folder = "data/spam"
    ham_folder = "data/ham"
    test_folder = "data/testing"
    
    # generate the file lists for training
    file_lists = []
    file_lists = [select_files(folder) for folder in (spam_folder, ham_folder)]
        
    # Learn the distributions    
    probabilities_by_category = learn_distributions(file_lists)
    
    # prior class distribution
    priors_by_category = [0.5, 0.5]
    
    # Store the classification results
    performance_measures = np.zeros([2,2])
    
    # explanation of performance_measures:
    # columns and rows are indexed by 0 = 'spam' and 1 = 'ham'
    # rows correspond to true label, columns correspond to guessed label
    # to be more clear, performance_measures = [[p1 p2]
    #                                           [p3 p4]]
    # p1 = Number of emails whose true label is 'spam' and classified as 'spam' 
    # p2 = Number of emails whose true label is 'spam' and classified as 'ham' 
    # p3 = Number of emails whose true label is 'ham' and classified as 'spam' 
    # p4 = Number of emails whose true label is 'ham' and classified as 'ham' 
    
    # Classify emails from testing set and measure the performance
    for filename in (util.get_files_in_folder(test_folder)):
        # Classify
        label,log_posterior = classify_new_email(filename,
                                                 probabilities_by_category,
                                                 priors_by_category)
        
        # Measure performance (the filename indicates the true label)
        base = os.path.basename(filename)
        true_index = ('ham' in base) 
        guessed_index = (label == 'ham')
        performance_measures[int(true_index), int(guessed_index)] += 1
    
    # Print the final error counts
    print(f"Number of Type 1 errors (False Negatives): {int(performance_measures[0,1])}")
    print(f"Number of Type 2 errors (False Positives): {int(performance_measures[1,0])}")
    
    template="You correctly classified %d out of %d spam emails, and %d out of %d ham emails."
    # Correct counts are on the diagonal
    correct = np.diag(performance_measures)
    # totals are obtained by summing across guessed labels
    totals = np.sum(performance_measures, 1)
    print(template % (correct[0],totals[0],correct[1],totals[1]))
    
    ### TODO: Write your code here to modify the decision rule such that
    ### Type 1 and Type 2 errors can be traded off, plot the trade-off curve
    generate_tradeoff_plot(test_folder, probabilities_by_category)
