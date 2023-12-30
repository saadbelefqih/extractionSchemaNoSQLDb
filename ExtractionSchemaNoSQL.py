#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install torch==1.2.0 torchvision==0.4.0 -f https://download.pytorch.org/whl/torch_stable.html')
get_ipython().system('pip install -U sentence-transformers')
# import libraries MySQL integration
get_ipython().system('pip install PyMySQL')
get_ipython().system('pip install ipython-sql')
get_ipython().system('pip install mysqlclient')
get_ipython().system('pip install scikit-learn numpy')
get_ipython().system('pip install matplotlib seaborn')


# In[2]:


# Create a Python class
class Triplet:
  def __init__(self,id,entity,property_name,dataType, embedding):
    self.id = id
    self.entity = entity
    self.property_name = property_name
    self.dataType = dataType
    self.embedding = embedding


  def __str__(self):
      return f"{self.id}{self.entity}{self.property_name}{self.dataType}{self.embedding}"


# In[4]:


def extract_triplets(json_obj, entity_name=None, parent_property=None):
    triplets = set()
    if isinstance(json_obj, dict):
        for key, value in json_obj.items():
            subject = entity_name if entity_name else ""
            property_name = key

            if isinstance(value, (str, int, float, bool)):
                data_type = type(value).__name__
                triplets.add((subject, property_name, data_type))
            elif isinstance(value, dict):
                if subject:
                    triplets.add((subject, "hasObject", property_name))
                nested_triplets = extract_triplets(value, entity_name=key, parent_property=property_name)
                if nested_triplets:
                    triplets.update(nested_triplets)
            elif isinstance(value, list) and len(value) > 0:
                if subject:
                    triplets.add((subject, "hasArray", property_name))
                nested_triplets = extract_triplets(value[0], entity_name=key, parent_property=property_name)
                if nested_triplets:
                    triplets.update(nested_triplets)
        return triplets


# In[28]:


import json
import os
# import pandas library
import pandas as pd

# Specify the directory where the JSON files are extracted
json_directory = './mongodb/'

# Create an empty list to store the JSON data
#data = []
all_triplets= set()

# List all files in the directory
json_files = [f for f in os.listdir(json_directory) if f.endswith('.json')]

# Loop through the JSON files and read each one
for json_file in json_files:
    # Specify the path to the JSON file
    json_file_path = os.path.join(json_directory, json_file)
    entity_name = os.path.basename(json_file_path).split('.')[0]
    # Read the JSON file and append its data to the list
    with open(json_file_path, 'r', encoding="utf8") as file:
        for line in file:
            json_obj =json.loads(line)
            extracted_triplets = extract_triplets(json_obj,entity_name=entity_name)
            all_triplets.update(extracted_triplets)

print(f"size of all_triplets{len(all_triplets)}")


# In[29]:


for triplet1 in list(all_triplets):
    print(f":{triplet1[0]} : {triplet1[1]}, : {triplet1[2]}")


# In[30]:


import pymysql


# In[31]:


db_name = "extractionschema"
db_host = "localhost"
db_username = "root"
db_password = ""

try:
    conn = pymysql.connect(host = db_host,
                           port = int(3306),
                           user = "root",
                           password = db_password,
                           db = db_name)
except e:
    print (e)
if conn:
    print ("connection successful")
else:
        print ("error")


# In[32]:


acronyms = pd.read_sql_query("select * from acronyms", conn)
acronyms=acronyms.values
for idAcronyms,shortForm,fullForm in acronyms:
    print(f"shortForm=>{shortForm}")


# In[33]:


import zipfile
zip_file_name = './model.zip'
extraction_path = './model'
model_save_path = './model/output/model'
# Unzip the file
with zipfile.ZipFile(zip_file_name, 'r') as zip_ref:
    zip_ref.extractall(extraction_path)


# In[34]:


from transformers import BertTokenizer
from sentence_transformers import SentenceTransformer
import torch
import re
from sklearn.metrics.pairwise import cosine_similarity
# Initialize a BERT model and tokenizer
#model_name = "bert-base-uncased"  # You can choose a different BERT model
#tokenizer = BertTokenizer.from_pretrained(model_name)
model = SentenceTransformer(model_save_path)
#model = BertModel.from_pretrained(model_name)


# In[35]:


# Function to convert triplet to BERT embeddings
def triplet_to_embeddings(entity, property_name, data_type):
    # Combine the tokens into a single list for the triplet
    combined_tokens = entity+' ' + property_name+' ' + data_type
    print(f"#{combined_tokens}")
    # Identifier normalization: Convert tokens to lowercase and remove numeric values
    words = re.findall(r'[a-z]+|[A-Z][a-z]*', combined_tokens)
    # Convert the words to lowercase using a list comprehension
    triplet_preprocessing = [token.lower() for token in words]
    # Acronym handling: Replace acronyms with their full terms
    for i, token in enumerate(triplet_preprocessing):
        for idAcronyms,shortForm,fullForm in acronyms:
            if shortForm == triplet_preprocessing[i]:
                triplet_preprocessing[i]=fullForm
    # Combine the triplet elements into a single string
    triplet_preprocessing_text = " ".join(triplet_preprocessing)
    print(f"*{triplet_preprocessing_text}")
    # Calculate the mean of embeddings along the sequence dimension
    triplet_embedding = model.encode(triplet_preprocessing_text, convert_to_tensor=True)
    return triplet_embedding


# In[36]:


triplets_embeddings =set()
# Loop through the list of triplets for i,triplet in enumerate(list(all_triplets)[:20]):

for i,triplet in enumerate(list(all_triplets)):
    entity, property_name, data_type = triplet
    # Calculate BERT embeddings for the current triplet
    if data_type in ['str', 'int', 'float', 'bool']:
        embedding = triplet_to_embeddings(entity, property_name, "")
        triplet= Triplet(i,entity,property_name,data_type,embedding)
        triplets_embeddings.add(triplet)
    elif property_name in ["hasObject","hasArray"]:
        embedding = triplet_to_embeddings(entity, "", data_type)
        triplet= Triplet(i,entity,property_name,data_type,embedding)
        triplets_embeddings.add(triplet)


# In[37]:


print(f"#{(len(triplets_embeddings))}")
for i,triplet1 in enumerate(list(triplets_embeddings)):
    print(f":{i}=> {triplet1.id},{triplet1.entity} : {triplet1.property_name}, : {triplet1.dataType}, : {triplet1.embedding}")


# In[38]:


cursor = conn.cursor()

# Define the table name from which you want to remove all data
table_name = 'triplets'  # Replace with your actual table name

# Remove all data from the table using the DELETE statement
delete_query = f"DELETE FROM {table_name}"
cursor.execute(delete_query)
conn.commit()

for i,triplet in enumerate(triplets_embeddings):
    sql = f"INSERT INTO {table_name} (id,entity, property_name, dataType, embedding) VALUES (%s,%s, %s, %s, %s)"
    values = (i,triplet.entity, triplet.property_name, triplet.dataType, triplet.embedding)

    cursor.execute(sql, values)
    conn.commit()

cursor.close()
#conn.close()


# In[39]:


#from sklearn.metrics.pairwise import cosine_similarity
from torch.utils.data import DataLoader
import torch
from sentence_transformers import util


# In[40]:


def eliminate_similar_triplets(triplets, threshold):
    # Create a set to store the indices of triplets to remove
    to_remove = set()
    similar_triplets = []
    filtered_triplets= set()
    for i in range(len(triplets)):
        for j in range(i + 1, len(triplets)):
            triplet1 = triplets[i]
            triplet2 = triplets[j]
            cosine_sim = util.pytorch_cos_sim(triplet1.embedding, triplet2.embedding)
            if cosine_sim.item() >= threshold:
                similar_triplets.append((triplet1, triplet2, cosine_sim))
                if triplet2.dataType in ['str', 'int', 'float', 'bool']:
                    to_remove.add(triplet2)
                else:
                    to_remove.add(triplet1)


    new_list = list(set(triplets).difference(to_remove))
    return new_list,to_remove,similar_triplets


# In[41]:


threshold = 0.60  # Define your cosine similarity threshold
triplets_embeddings1 = list(triplets_embeddings)
filtered_triplets,to_remove,similar_triplets = eliminate_similar_triplets(triplets_embeddings1, threshold)
filtered_triplets = [triplets_embeddings1[i] for i in range(len(triplets_embeddings1)) if i not in to_remove]


# In[42]:


print(f"#original {(len(triplets_embeddings))}")
print(f"#filtered_triplets {(len(filtered_triplets))}")
print(f"#to_remove {(len(to_remove))}")
print(f"*********************************************************************")
for triplet1, triplet2, cosine_sim in similar_triplets:
    print(f": {triplet1.id},{triplet1.entity} : {triplet1.property_name}, : {triplet1.dataType}")
    print(f": {triplet2.id},{triplet2.entity} : {triplet2.property_name}, : {triplet2.dataType}")
    print(f"Similarity: {cosine_sim.item()}\n")
print(f"*********************************************************************")
for i,triplet1 in enumerate(list(filtered_triplets)):
    print(f":{i}=> {triplet1.id},{triplet1.entity} : {triplet1.property_name}, : {triplet1.dataType}")


# In[43]:


import networkx as nx
import matplotlib.pyplot as plt


# In[49]:


filtered_triplets=list(filtered_triplets)
# Create an empty directed graph to represent the tree structure
G = nx.DiGraph()

# Iterate through filtered_triplets
for triplet in filtered_triplets:
    if triplet.property_name in ('hasObject', 'hasArray'):
        # Create nodes for entity and third part of triplet
        G.add_node(triplet.entity, node_type="entity")
        G.add_node(triplet.dataType, node_type="entity")

        # Connect them with labeled edges
        G.add_edge(triplet.entity, triplet.dataType, label=triplet.property_name, node_type="property")
    else:
        # Create a simple triplet with entity, property, and leaf
        G.add_node(triplet.entity, node_type="entity")
        G.add_node(triplet.dataType, node_type="leaf")
        G.add_edge(triplet.entity, triplet.dataType, label=triplet.property_name, node_type="property")

# Create separate lists for nodes of each type
entity_nodes = [node for node, node_data in G.nodes(data=True) if node_data.get("node_type") == "entity"]
property_nodes = [node for node, node_data in G.nodes(data=True) if node_data.get("node_type") == "property"]
leaf_nodes = [node for node, node_data in G.nodes(data=True) if node_data.get("node_type") == "leaf"]

# Plot the graph with different node colors
plt.figure(figsize=(30, 20))  # You can adjust the size as needed
pos = nx.spring_layout(G, k=3)  # You can use a different layout if needed
edge_labels = {(entity1, entity2): G[entity1][entity2]["label"] for entity1, entity2 in G.edges()}
nx.draw_networkx_nodes(G, pos, nodelist=entity_nodes, node_color="lightblue", node_size=9999)
nx.draw_networkx_nodes(G, pos, nodelist=property_nodes, node_color="lightgreen", node_size=999)
nx.draw_networkx_nodes(G, pos, nodelist=leaf_nodes, node_color="lightcoral", node_size=5999)
nx.draw_networkx_edges(G, pos)
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels,font_size=16)
nx.draw_networkx_labels(G, pos)
# Save the figure in a higher quality format
plt.savefig("graph_output.png", dpi=300, bbox_inches="tight")
plt.show()


# In[2]:


import matplotlib.pyplot as plt

def calculate_precision_recall_f1(true_positive, false_positive, false_negative):
    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
    recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return precision, recall, f1_score

# Example usage:
true_positive = 50
false_positive = 0
false_negative = 10

precision, recall, f1_score = calculate_precision_recall_f1(true_positive, false_positive, false_negative)

# Plotting
labels = ['Precision', 'Recall', 'F1 Score']
values = [precision, recall, f1_score]

plt.bar(labels, values, color=['lightblue', 'lightgreen', 'lightcoral'])
plt.title('Precision, Recall, and F1 Score')
plt.ylabel('Score')
plt.ylim(0, 1)  # Set y-axis limit to 0-1 for better visualization
# Save the figure in a higher quality format
plt.savefig("precision_recall_f1.png", dpi=300, bbox_inches="tight")
plt.show()


# In[40]:





# In[45]:


from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Assuming you have a list of Triplets named triplets_list
# Each Triplet object has an 'embedding' attribute containing the embedding vector
triplets_list=triplets_embeddings1
# Extract embeddings from Triplets
embeddings = [triplet.embedding.cpu().detach().numpy() for triplet in triplets_list]

# Convert the list of embeddings into a numpy array
embedding_array = np.array(embeddings)

# Calculate the cosine similarity matrix
similarity_matrix = cosine_similarity(embedding_array)

# Set the labels for the axes
labels = [triplet.id for triplet in triplets_list]

# Create a heatmap using seaborn
sns.heatmap(similarity_matrix, xticklabels=labels, yticklabels=labels, cmap="YlGnBu")

# Add labels and show the plot
plt.title("Similarity Matrix Heatmap")
plt.xlabel("Triplets")
plt.ylabel("Triplets")
plt.savefig("similarity_matrix.png", dpi=300, bbox_inches="tight")
plt.show()


# In[ ]:




