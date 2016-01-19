from collections import Counter, defaultdict
from bs4 import BeautifulSoup # for parsing the html
import sys, os, re

# breaks text into words
def text_split(text):
    return re.findall('[a-z0-9]+', text)

# train classifier with training data 
def train_classifier(positive_count, negative_count):
    priors = Counter()
    likelihood = defaultdict(Counter)
    total_positive_training_words = 0 
    total_negative_training_words = 0

    with open('pos-train.txt', 'r') as pos:
        data = pos.read().decode('utf8')
        priors['positive'] = positive_count
        for word in text_split(data):
            total_positive_training_words += 1
            likelihood['positive'][word] += 1

    with open('neg-train.txt', 'r') as neg:
        data = neg.read().decode('utf8')
        priors['negative'] = negative_count
        for word in text_split(data):
            total_negative_training_words += 1
            likelihood['negative'][word] += 1

    return(priors, likelihood)
    # return (priors, likelihood, total_positive_training_words, total_negative_training_words)

# classifier - uses priors & likelihood to calculate the posterior with test data  
def classify_bayesian(filename, priors, likelihood):
    # return the class that maximizes the posterior
    max_class = (-1E6, '')
    with open(filename, 'r') as f:
        data = f.read()
        for category in priors.keys():
            # the number of articles in that category (used to estimate category frequency) 
            p = priors[category]

            # calculate the total number of words for that category (positive or negative)
            # used to estimate word frequency in a category
            n = float(sum(likelihood[category].itervalues()))

            # calculate the posterior by summing the products of word frequency and category 
            # frequency for each word
            # 10,000 is an arbitrary value I found I could scale by such that the least number of
            # "infinite" and "0.0" p-values would be printed and highest accuracy would occur. 
            for word in text_split(data):
                p = 10000 * p * max(1E-10, likelihood[category][word] / n)

            # used for debugging 
            # print p

            if p > max_class[0]:
                max_class = (p,category)
        return max_class[1]

# recursively print all the p-elements following the passed in element until at least one is found
def print_details(sibling, details):
    if sibling.name == 'p':
        details += sibling.text + "\n"
        if sibling.next_sibling and sibling.next_sibling.next_sibling:
            return print_details(sibling.next_sibling.next_sibling, details)
        else:
            if details:
                return details
            else:
                details += "Couldn't find any relevant information\n"
                return details
    else:
        if details:
            return details
        else:
            if sibling.next_sibling and sibling.next_sibling.next_sibling:
                return print_details(sibling.next_sibling.next_sibling, details)
            else:
                details += "Couldn't find any relevant information\n"
                return details


def main():
    # directories to train and test classifier
    positive_train_path = sys.argv[1]
    negative_train_path = sys.argv[2]
    positive_test_path = sys.argv[3]
    negative_test_path = sys.argv[4]

    # obtain files in directories 
    positive_train_pages = os.listdir(positive_train_path)
    negative_train_pages = os.listdir(negative_train_path)
    positive_test_pages = os.listdir(positive_test_path)
    negative_test_pages = os.listdir(negative_test_path)

    positive_page_count = 0
    negative_page_count = 0

    # open new files to load training data into
    positive_train_file = open('pos-train.txt', 'w')
    positive_train_file.seek(0)
    positive_train_file.truncate()
    negative_train_file = open('neg-train.txt', 'w')
    negative_train_file.seek(0)
    negative_train_file.truncate()

    # load training data for disease related articles 
    for page in positive_train_pages:
        if page.startswith('.'):
            continue 
        page = "training/positive/" + page
        filename = open(page, 'r')
        positive_page_count += 1
        soup = BeautifulSoup(filename, 'html.parser')
        for p in soup.find_all('p'):
            positive_train_file.write((p.text).encode('utf8'))
        filename.close()

    # load training data for non-disease related articles 
    for page in negative_train_pages:
        if page.startswith('.'):
            continue
        page = "training/negative/" + page
        filename = open(page)
        negative_page_count += 1
        soup = BeautifulSoup(filename, 'html.parser')
        for p in soup.find_all('p'):
            negative_train_file.write((p.text).encode('utf8'))
        filename.close()

    positive_train_file.close()
    negative_train_file.close()


    (priors, likelihood) = train_classifier(positive_page_count, negative_page_count)

    ############################
    # for debugging purposes 

    # build prior and likelihood data to train classifier 
    # (priors, likelihood, total_positive_training_words, 
    #     total_negative_training_words) = train_classifier(positive_page_count, 
    #                                                         negative_page_count)

    # print priors
    # print total_positive_training_words
    # print total_negative_training_words
    # print likelihood
   #############################

    num_correct = 0
    total_test_pages = 0

    # open file to report information about diseases that were correctly classified 
    pertinent_data = open('pertinent_data.txt', 'w')
    pertinent_data.seek(0)
    pertinent_data.truncate()
    pertinent_data.write("THE FOLLOWING DISEASES WERE CORRECTLY CLASSIFIED (supplementary information provided) :\n")
    # test classifier with disease-related articles
    for page in positive_test_pages:
        if page.startswith('.'):
            continue
        total_test_pages += 1
        page = "testing/positive/" + page
        filename = open(page)
        soup = BeautifulSoup(filename, 'html.parser')

        # open new files to load testing data into
        positive_test_file = open('pos-test.txt', 'w')
        positive_test_file.seek(0)
        positive_test_file.truncate()

        for p in soup.find_all('p'):
            positive_test_file.write((p.text).encode('utf8'))
        filename.close()
        positive_test_file.close()

        # record data 
        if classify_bayesian('pos-test.txt', priors, likelihood) == 'positive':
            num_correct += 1
            title = soup.title.text.split(' -', 1)[0]
            pertinent_data.write((title.upper() + "\n").encode('utf8'))
            found_data = 0
            for span in soup.find_all('span', {'class' : 'mw-headline'}):
                if "symptoms" in span.text.lower():
                    pertinent_data.write("\tSYMPTOMS:\n")
                    details = print_details(span.parent.next_sibling.next_sibling, "")
                    pertinent_data.write(details.encode('utf8'))
                    found_data = 1
                elif "diagnosis" in span.text.lower():
                    pertinent_data.write("\tDIAGNOSIS:\n")
                    details = print_details(span.parent.next_sibling.next_sibling, "")
                    pertinent_data.write(details.encode('utf8'))
                    found_data = 1
                elif "causes" in span.text.lower():
                    pertinent_data.write("\tCAUSES:\n")
                    details = print_details(span.parent.next_sibling.next_sibling, "")
                    pertinent_data.write(details.encode('utf8'))
                    found_data = 1
                elif "treatment" in span.text.lower():
                    pertinent_data.write("\tCAUSES:\n")
                    details = print_details(span.parent.next_sibling.next_sibling, "")
                    pertinent_data.write(details.encode('utf8'))
                    found_data = 1
                elif "prognosis" in span.text.lower():
                    pertinent_data.write("\tPROGNOSIS:\n")
                    details = print_details(span.parent.next_sibling.next_sibling, "")
                    pertinent_data.write(details.encode('utf8'))
                    found_data = 1
            if found_data == 0:
                pertinent_data.write("\tSorry wasn't able to scrape any relevant data\n")

    pertinent_data.close()
        
    # test classifier with non-disease related articles 
    for page in negative_test_pages:
        if page.startswith('.'):
            continue
        total_test_pages += 1
        page = "testing/negative/" + page
        filename = open(page)
        soup = BeautifulSoup(filename, 'html.parser')

        # open new files to load testing data into
        negative_test_file = open('neg-test.txt', 'w')
        negative_test_file.seek(0)
        negative_test_file.truncate()

        for p in soup.find_all('p'):
            negative_test_file.write((p.text).encode('utf8'))
        filename.close()
        negative_test_file.close()

        # record data 
        if classify_bayesian('neg-test.txt', priors, likelihood) == 'negative':
            num_correct += 1

    # report data
    print "\n"
    print "Classified {0} correctly out of {1}; {2:.2f}%% correct\n".format(num_correct, total_test_pages,
        100 * float(num_correct)/total_test_pages)

if __name__ == '__main__':
    main()