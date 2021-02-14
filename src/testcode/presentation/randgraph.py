# Random graph for the picture for the presentation

import networkx as nx
import matplotlib.pyplot as plt




G=nx.fast_gnp_random_graph(100,0.05,seed=17)
pos=nx.spring_layout(G)

nodes=nx.draw_networkx_nodes(G, pos,node_size=30,node_color="#aaaaaa",alpha=0.8,linewidths=0.1)
edges=nx.draw_networkx_edges(G, pos,alpha=0.1)
plt.show()