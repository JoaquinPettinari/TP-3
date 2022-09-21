import pandas as pd #for manipulating the csv data
import numpy as np #for mathematical calculation

def calc_total_entropy(train_data, label, class_list):
    total_row = train_data.shape[0]
    total_entr = 0
    for c in class_list:
        total_class_count = train_data[train_data[label] == c].shape[0]
        total_class_entr = - (total_class_count/total_row)*np.log2(total_class_count/total_row) 
        total_entr += total_class_entr
    
    return total_entr

def calc_entropy(feature_value_data, label, class_list):
    class_count = feature_value_data.shape[0]
    entropy = 0
    
    for c in class_list:
        label_class_count = feature_value_data[feature_value_data[label] == c].shape[0]
    
        entropy_class = 0
        if label_class_count != 0:
            probability_class = label_class_count/class_count
            entropy_class = - probability_class * np.log2(probability_class) 
        
        entropy += entropy_class
        
    return entropy

def calc_info_gain(feature_name, train_data, label, class_list):
    feature_value_list = train_data[feature_name].unique()
    total_row = train_data.shape[0]
    feature_info = 0.0
    
    for feature_value in feature_value_list:
        feature_value_data = train_data[train_data[feature_name] == feature_value]
        feature_value_count = feature_value_data.shape[0]
        feature_value_entropy = calc_entropy(feature_value_data, label, class_list)
        feature_value_probability = feature_value_count/total_row
        feature_info += feature_value_probability * feature_value_entropy
        
    return calc_total_entropy(train_data, label, class_list) - feature_info

def find_most_informative_feature(train_data, label, class_list):
    feature_list = train_data.columns.drop(label)
    max_info_gain = -1
    max_info_feature = None
    
    for feature in feature_list:  
        feature_info_gain = calc_info_gain(feature, train_data, label, class_list)
        if max_info_gain < feature_info_gain:
            max_info_gain = feature_info_gain
            max_info_feature = feature
            
    return max_info_feature

def generate_sub_tree(feature_name, train_data, label, class_list):
    feature_value_count_dict = train_data[feature_name].value_counts(sort=False)
    tree = {}
    
    for feature_value, count in feature_value_count_dict.iteritems():
        feature_value_data = train_data[train_data[feature_name] == feature_value]
        
        assigned_to_node = False
        for c in class_list:
            class_count = feature_value_data[feature_value_data[label] == c].shape[0]

            if class_count == count:
                tree[feature_value] = c
                train_data = train_data[train_data[feature_name] != feature_value]
                assigned_to_node = True
        if not assigned_to_node:
            tree[feature_value] = "?"
            
    return tree, train_data

def make_tree(root, prev_feature_value, train_data, label, class_list):
    print("Root: ", root)
    print("prev_feature_value: ",prev_feature_value)
    print("train_data: ", train_data)
    print("label: ", label)
    print("class_list: ", class_list)
    
    # Entra al if si hay datos
    if train_data.shape[0] != 0:
        print("Shape: ", train_data.shape)
        # Trae próximo atributo padre
        max_info_feature = find_most_informative_feature(train_data, label, class_list)
        print("Max info: ", max_info_feature)
        tree, train_data = generate_sub_tree(max_info_feature, train_data, label, class_list)
        print("Tree: ", tree)
        print("Train Data: ", train_data)
        next_root = None
        if(len(train_data) == 0):
            return root
        
        if prev_feature_value != None:
            root[prev_feature_value] = dict()
            print("New dict", root[prev_feature_value])
            root[prev_feature_value][max_info_feature] = tree
            print("Arbol en dict", root[prev_feature_value][max_info_feature])
            next_root = root[prev_feature_value][max_info_feature]
            print("Next root: ", next_root)
            
        else:
            # Concatena el los valores del atributo padre con el arbol
            root[max_info_feature] = tree
            print("Root en max info:" , root)
            next_root = root[max_info_feature]
            print("Next Root: ", next_root)
        
        for node, branch in list(next_root.items()):
            print("Branch: ", branch)
            print("Node: ", node)
            
            if branch == "?":
                feature_value_data = train_data[train_data[max_info_feature] == node]
                print("FEATURE: ")
                print(feature_value_data)
                make_tree(next_root, node, feature_value_data, label, class_list)

def id3(train_data_m, label):
    train_data = train_data_m.copy()
    tree = {}
    class_list = train_data[label].unique()
    make_tree(tree, None, train_data_m, label, class_list)
    
    return tree

def predict(tree, instance):
    if not isinstance(tree, dict):
        return tree
    else:
        root_node = next(iter(tree))
        feature_value = instance[root_node]
        if feature_value in tree[root_node]:
            return predict(tree[root_node][feature_value], instance)
        else:
            return None

def evaluate(tree, test_data_m, label):
    actualCreditabilities = []
    for index, row in test_data_m.iterrows():
        result = predict(tree, test_data_m.iloc[index])
        actualCreditabilities.append(result)        
    return actualCreditabilities

def imprimir_arbol(arbol):
    for x in arbol:
        print (x)
        for y in arbol[x]:
            print (y,':',arbol[x][y])
