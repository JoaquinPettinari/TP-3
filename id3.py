import pandas as pd
import math
from collections import Counter
from nodo import Nodo
import matplotlib.pyplot as plt
from pprint import pprint

class ID_3:
    def __init__(self, df, clasePrimaria, atributos):
        # print("\n Given Play Golf Dataset:\n\n", df)
        # print('Target Attribute is   ➡ ', clasePrimaria)
        self.df = df
        self.clasePrimaria = clasePrimaria
        self.atributos = atributos
        self.raiz = None
        self.arbol = self.aplicar_identificador(df, clasePrimaria, atributos, self.raiz)
       

    def entropia(self, prob):
        # Calculo de la entropia recibe lista con 2 valores y aplica fórmula de -sum prob*log(prob)
        return sum( [-prob*math.log(prob, 2) for prob in prob])    

    def aplicar_identificador(self, df, target_attribute, attribute_names, nodo_actual: Nodo, default_class=None):
        
        cnt = Counter(x for x in df[target_attribute])# class of YES /NO
        # Cuando hay un solo valor posible. Lo devuelve. Acá con los árboles deberíamos mirar si es hoja y devolverlo
        if len(cnt) == 1:
            return next(iter(cnt)) 
        
        ## Second check: Is this split of the dataset empty? if yes, return a default value
        # Todavía no entiendo que hace xd. Miré los logs pero no encontré nada
        elif df.empty or (not attribute_names):
            return default_class  # Return None for Empty Data Set
        
        ## Otherwise: This dataset is ready to be devied up!
        else:
            # Get Default Value for next recursive call of this function:
            default_class = max(cnt.keys()) #No of YES and NO Class
            # Compute the Information Gain of the attributes:
            gainz=[]
            for attr in attribute_names:
                ig= self.information_gain(df, attr, target_attribute)
                gainz.append(ig)
                #print('\nInformation gain of','“',attr,'”','is ➡', ig)
                #print("=========================================================")
            index_of_max = gainz.index(max(gainz))               # Index of Best Attribute
            best_attr = attribute_names[index_of_max]
            if(nodo_actual == None):
                self.raiz = Nodo(best_attr)
                nodo_actual = self.raiz
                      # Choose Best Attribute to split on
            #print(self.raiz.imprimirme())
            #print("\nList of Gain for arrtibutes:",attribute_names,"\nare:", gainz,"respectively.")
            #print("\nAttribute with the maximum gain is ➡", best_attr)
            #print("\nHence, the Root node will be ➡", best_attr)
            #print("=========================================================")

            # Create an empty tree, to be populated in a moment
            tree = {best_attr:{}} # Initiate the tree with best attribute as a node 
            remaining_attribute_names =[i for i in attribute_names if i != best_attr]
            
            # Split dataset-On each split, recursively call this algorithm.Populate the empty tree with subtrees, which
            # are the result of the recursive call
            print(nodo_actual.imprimirme())
            for attr_val, data_subset in df.groupby(best_attr):
                nodo_hijo = Nodo(attr_val)
                nodo_actual.agregar_hijo(nodo_hijo)
                subtree = self.aplicar_identificador(data_subset,target_attribute, remaining_attribute_names, nodo_hijo ,default_class,best_attr)
                print(nodo_actual.valor, subtree)
                tree[best_attr][attr_val] = subtree
            return tree
    
    def information_gain(self, df, split_attribute, target_attribute):
        #print("\n\n----- Information Gain Calculation of",split_attribute,"----- ") 
        
        # group the data based on attribute values
        df_split = df.groupby(split_attribute) 
        glist=[]
        for gname,group in df_split:
            #print('Grouped Attribute Values \n',group)
            #print("---------------------------------------------------------")
            glist.append(gname) 
        
        glist.reverse()
        nobs = len(df.index) * 1.0   
        df_agg1=df_split.agg({target_attribute:lambda x:self.entropy_of_list(x, glist.pop())})
        df_agg2=df_split.agg({target_attribute :lambda x:len(x)/nobs})
        
        df_agg1.columns=['Entropy']
        df_agg2.columns=['Proportion']
        
        # Calculate Information Gain:
        new_entropy = sum( df_agg1['Entropy'] * df_agg2['Proportion'])
        old_entropy = self.entropy_of_list(df[target_attribute])
        return old_entropy - new_entropy

    def entropy_of_list(self, ls):  
        
        # Total intances associated with respective attribute
        total_instances = len(ls)  # = 14
        #print("---------------------------------------------------------")
        #print("\nTotal no of instances/records associated with '{0}' is ➡ {1}".format(value,total_instances))
        # Counter calculates the propotion of class
        cnt = Counter(x for x in ls)
        #print('\nTarget attribute class count(Yes/No)=',dict(cnt))
       
        # x means no of YES/NO
        probs = [x / total_instances for x in cnt.values()]  
        # print("\nClasses➡", max(cnt), min(cnt))
        # print("\nProbabilities of Class 'p'='{0}' ➡ {1}".format(max(cnt),max(probs)))
        # print("Probabilities of Class 'n'='{0}'  ➡ {1}".format(min(cnt),min(probs)))
        
        # Call Entropy 
        return self.entropia(probs)

    def entropia_de_un_conjunto(self, conjunto):
        # Contador que calcula la frecuencia relativa de el conjunto Counter({'1': 9, '0': 5 })
        contador = Counter(x for x in conjunto)   
        # Numero total de datos (Ejemplo arriba es 14)
        num_instancias = len(conjunto)*1.0          
        
        # Se mapea el contador / el total para obtener la probabilidad. 9/14 y 5/14 -> [0.35714, 0.64285]
        probs = [x / num_instancias for x in contador.values()]        
        # Call Entropy
        return self.entropia(probs) 
    
    def imprimirArbol(self):
        print("\nThe Resultant Decision Tree is: ⤵\n")
        pprint(self.arbol)