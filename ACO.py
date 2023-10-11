import numpy as np
import matplotlib.pyplot as plt
import sys
IN_COLAB = 'google.colab' in sys.modules

class Catagorization:
    def __init__(self, num_ants, num_rules_converge, data, attributes, classes):
        self.num_ants = num_ants
        self.num_rules_converge = num_rules_converge
        self.data = data
        self.original_data = data.copy()
        self.attributes = attributes

        self.classes = classes
        self.heuristic = self.calc_heuristic()
        self.pharamones = self.init_pharamones()
        self.discovered_rules = []
        self.qualities = [[]]
        self.leftover_cases = 35
        self.init_ants() 
        
    def init_ants(self):
        min_cases = 5
        self.ants = [Ant(self.pharamones, self.heuristic, self.attributes, min_cases, self.classes) for i in range(self.num_ants)]
        
    def run_simulation(self, verbose = False, supress_output = False):
        while len(self.data) > self.leftover_cases:
            self.converged_rules = 1
            best_rule = []
            best_quality = -1
            consequent = None
            
            for j, ant in enumerate(self.ants):
                self.run_one_ant(ant)
                
                    
                if best_quality < 0 and set(self.ants[j-1].rule) == set(ant.rule) and ant.rule:
                    self.converged_rules += 1
                else:
                    self.converged_rules = 1
                if self.converged_rules == self.num_rules_converge:
                    print('rules have converged')
                    break
                if ant.quality > best_quality:
                    best_quality = ant.quality
                    best_rule = ant.rule
                    consequent = ant.consequent
            n = self.remove_relevant_cases(best_rule, consequent)
            assert(not self.find_relevant_cases(best_rule))
            self.discovered_rules.append((best_rule, consequent, best_quality, n))
            self.pharamones = self.init_pharamones()
            self.init_ants()
            self.qualities.append([])
            if verbose: print('remaining data: ', len(self.data))
        if not supress_output: print('simulation done. Remaining cases: ', len(self.data))
            
    def run_one_ant(self, ant):
        ant.pharamones = self.pharamones
        ant.add_terms(self.data)
        q = self.prune_ant(ant)
        consequent = self.calc_consequent(ant.rule)
        self.update_pharamones(ant.rule, q)
        ant.quality = q
        ant.consequent = consequent
        self.qualities[-1].append(q)
        
    def calc_heuristic(self):
        probs = {}
        heuristic = {}
        for index, i in enumerate(self.attributes.keys()):
            for j in self.attributes[i]:
                for k in self.classes:
                    a = [game for game in self.data if k == game[-1]]
                    b = [game for game in a if j == game[index]]
                    p = len(b)/len(self.data)
                    c = probs.get((index,j),[])
                    c.append(np.log2(p**p))
                    probs[(index,j)] = c
                heuristic[(index,j)] = -sum(probs[(index,j)])
        return heuristic

    def pick_best_rule(self):
        most_correct = 0
        best_ant = None
        for ant in self.ants:
            consequence, num_correct = self.calc_num_correct(ant.rule)
            ant.consequence = consequence
            if num_correct > most_correct:
                most_correct = num_correct
                best_ant = ant
        return most_correct, best_ant
    
    def find_relevant_cases(self, rule):
        relevant_cases = self.data.copy()
        for case in self.data:
            for term in rule:
                if case[term[0]] != term[1]:
                    relevant_cases.remove(case)
                    break
        return relevant_cases
    
    def remove_relevant_cases(self, rule, consequent):
        relevant_cases = self.find_relevant_cases(rule)
        count = 0
        for case in relevant_cases:
            self.data.remove(case)
            if case[-1] == consequent:
                count += 1
        return count
    def calc_consequent(self, rule):
        relevant_cases = self.find_relevant_cases(rule)
        classes = {}
        if not relevant_cases:
            relevant_cases = self.data
            
        for case in relevant_cases:
            classes[str(case[-1])] = classes.get(str(case[-1]), 0) + 1
        res = max(classes, key=classes.get)
        return res
    
    def prune_ant(self, ant):
        """Iteratively goes through the rulelist for an ant and sees which rules are not helping the quality of the rule"""
        n = len(ant.rule)
        if n == 0:
            return 0
        for iteration in range(n):
            max_delta_quality = 0
            best_new_rule = ant.rule
            consequent = self.calc_consequent(ant.rule)
            base_quality = self.calc_quality(ant.rule, consequent)
            
            for i, term in enumerate(ant.rule):
                new_rule = ant.rule[0:i] + ant.rule[i+1:]
                new_consequent = self.calc_consequent(new_rule)
                new_quality = self.calc_quality(new_rule, new_consequent)
                if max_delta_quality <= new_quality - base_quality:
                    max_delta_quality = new_quality - base_quality
                    best_new_rule = new_rule
            if max_delta_quality >= 0:
                if best_new_rule:
                    ant.rule = best_new_rule
                    max_delta_quality = 0
            else:
                break
        return max_delta_quality + base_quality
        
    def calc_quality(self, rule, consequent):
        relevant_cases = self.find_relevant_cases(rule)
        num_cases_covered = len(relevant_cases)
        num_cases_not_covered = len(self.data) - len(relevant_cases)
        
        true_positives = len([case for case in relevant_cases if case[-1] == consequent])
        false_positives = len(relevant_cases) - true_positives
        false_negatives = len([case for case in self.data if case not in relevant_cases and case[-1] == consequent])
        true_negatives = len(self.data) - len(relevant_cases) - false_negatives
        
        sensitivity = true_positives / (true_positives + false_negatives)
        if true_negatives == 0: return 0
        specificity = true_negatives / (true_negatives + false_positives)
        
        return sensitivity * specificity
        
    def init_pharamones(self):
        pharamones = {}
        total = 0
        for attribute in self.attributes.keys():
            total += len(self.attributes[attribute])
            
        initial_value = 1/total
        for index, i in enumerate(self.attributes.keys()):
            for j in self.attributes[i]:
                pharamones[(index,j)] = initial_value
        return pharamones
    
    def update_pharamones(self, rule, quality):
        for term in rule:
            self.pharamones[(term[0], term[1])] += quality
        normalization_factor = sum(self.pharamones.values())
        keys = self.pharamones.keys()
        for key in keys:
            self.pharamones[key] /= normalization_factor
        
        assert(sum(self.pharamones.values()) - 1 < 0.0001)
        
    def evaluate_discovered_rules(self, data):
        num_correct = 0
        num_total = len(data)
        self.data = data
        for rule in self.discovered_rules:
            num_correct += self.remove_relevant_cases(rule[0], rule[1])
        return num_correct / num_total

class Ant:
    def __init__(self,pharamones, heuristic, attributes, min_cases_per_rule, classes):
        ""
        self.rule = []
        self.pharamones = pharamones
        self.heuristic = heuristic
        self.decision = np.zeros((9,3))
        self.k = 2
        self.classes = classes
        self.attributes = attributes
        self.min_cases_per_rule = min_cases_per_rule
        self.used_attributes = []
        
    def add_terms(self, data):
        "adds a term to the ruleset based on the pharamone trail and heuristic function"
        quit = False
        tries = 0
        for i in self.attributes.keys():
            probs = self.calc_prob()
            picking = True
            while picking:
                term = self.pick_term(probs)
                if len(self.find_relevant_cases(self.rule+[term], data)) > self.min_cases_per_rule:
                    if term[0] not in self.used_attributes:
                        self.used_attributes.append(term[0])
                        picking = False
                else:
                    tries += 1
                if tries > 10 and self.rule:
                    picking = False
                    quit = True
                if tries > 25:
                    quit = True
            if not quit:
                self.rule.append(term)
            else:
                break
    def normalize(self, function):
        norm = {}
        for index, i in enumerate(self.attributes.keys()):
            for j in self.attributes[i]:
                num = function(index, j)
                if num == 0:
                    norm[(index,j)] = 0
                    continue
                unused_attributes = len(self.attributes.keys())-len(self.rule)
                normalization_factor = 0
                for jj in self.attributes[i]:
                    normalization_factor += function(index, jj)
                norm[(index,j)] = num / (unused_attributes * normalization_factor)
        return norm
        
    def normalize_heuristic(self):
        def f(i, j):
            return np.log2(self.k) - self.heuristic[(i,j)]
        return self.normalize(f)
    
    def calc_prob(self):
        self.normalized_heuristic = self.normalize_heuristic()
        def f(i, j):
            if i not in self.used_attributes:
                return self.pharamones[(i,j)] * self.normalized_heuristic[(i,j)]
            else:
                return 0
        return self.normalize(f)
    
    def pick_term(self, probs):
        index = np.random.choice(len(probs), 1, p = list(probs.values()))
        index = index[0]
        term = list(probs.keys())[index]
        return term
    
    def find_relevant_cases(self, rule, data):
        relevant_cases = data.copy()
        for case in data:
            for term in rule:
                if case[term[0]] != term[1]:
                    relevant_cases.remove(case)
                    break
        return relevant_cases

import os
import string

def get_train_and_test_from_file(url, file, special = None):

    !wget -cq $url
    data = open(file, 'r')
    data_listed = []

    for i in data.readlines():
        if not special:
            data_listed.append(tuple(str.split(i[:-1], ',')))
        elif special == 'breast':
            line = str.split(i[:-1], ',')
            c = line.pop(0)
            line.append(c)
            data_listed.append(tuple(line))
        elif special == 'wisconsin':
            data_listed.append(tuple(str.split(i[:-1], ',')[1:]))
    n=len(data_listed)
    train_ind = np.random.choice(n, round(n*0.8), replace=False)
    train = [data_listed[index] for index in train_ind]
    test = [case for case in data_listed if case not in train]
    return train, test

"""### Running the Miner on Breast Cancer Data (Wisconsin)"""

# Number of times to run the miner
num_runs = 1
# Number of ants to use
num_ants = 50

accuracies = []
num_rules = []

for i in range(num_runs):
    
    train, test = get_train_and_test_from_file('/content/sample_data/Breast_Cancer.csv', '/content/sample_data/Breast_Cancer.csv', 'wisconsin')
    attribute_list = ['Clump Thickness', 'Uniformity of Cell Size', 'Uniformity of Cell Shape', 'Marginal Adhesion', 'Single Epithelial Cell Size', 'Bare Nuclei', 'Bland Chromatin', 'Normal Nucleoli', 'Mitoses']
    attributes = {}
    for attribute in attribute_list:
        attributes[attribute] = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']

    classes = ['2', '4']

    s = Catagorization(num_ants, 10, train, attributes, classes)
    s.run_simulation()
    accuracies.append(s.evaluate_discovered_rules(test))
    num_rules.append(len(s.discovered_rules))

print('Average Accuracy is: ', np.mean(accuracies))
print('Average number of features per ruleset is: ', np.mean(num_rules))
print('Number of rules per ruleset std. is', np.mean(num_rules))

"""### Sweeping number of Ants
"""

# Number of times to run the miner
num_runs = 15
# Number of ants to use
num_ants_vec = [5, 10, 15, 20, 25]

all_accuracies = []
all_num_rules = []
all_num_terms = []
for num_ants in num_ants_vec:
    all_accuracies.append([])
    all_num_rules.append([])
    all_num_terms.append([])
    
    for i in range(num_runs):
        train, test = get_train_and_test_from_file('/content/sample_data/Breast_Cancer.csv', '/content/sample_data/Breast_Cancer.csv', 'wisconsin')
        attribute_list = ['Clump Thickness', 'Uniformity of Cell Size', 'Uniformity of Cell Shape', 'Marginal Adhesion', 'Single Epithelial Cell Size', 'Bare Nuclei', 'Bland Chromatin', 'Normal Nucleoli', 'Mitoses']
        attributes = {}
        for attribute in attribute_list:
            attributes[attribute] = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']

        classes = ['2', '4']

        s = Catagorization(num_ants, 10, train, attributes, classes)
        s.run_simulation(supress_output = True)
        all_accuracies[-1].append(s.evaluate_discovered_rules(test))
        all_num_rules[-1].append(len(s.discovered_rules))
        all_num_terms[-1].append(np.mean([len(rule[1]) for rule in s.discovered_rules]))
        
plt.boxplot(all_accuracies, positions=[n/5 for n in num_ants_vec])
plt.title('Accuracy vs Number of Ants Used (Wisconsin Breast Cancer)')
plt.xlabel('Number of Ants Used (x5)')
plt.ylabel('Accuracy of Ruleset')
plt.show()

