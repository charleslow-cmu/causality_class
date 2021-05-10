class MixedGraph:
    def __init__(self):
        self.nodes = set()
        self.dirEdges = set()
        self.undirEdges = set()
        self.nodeLevel = {}
        
    def addNode(self, V, level=0):
        self.nodes.add(V)
        self.nodeLevel[V] = level
    
    # 0 for undirected, 1 for directed
    def addEdge(self, u, v, type=0):
        if type == 0:
            self.undirEdges.add(frozenset([u, v]))
        
        elif type == 1:
            self.dirEdges.add((u, v))
            
    def edges(self, V, type=0):
        edgelist = []
        if type == 0:
            for edge in self.undirEdges:
                if V in edge:
                    edgelist.append(edge)
        
        if type == 1:
            for edge in self.dirEdges:
                if V in edge:
                    edgelist.append(edge)
        return edgelist
    
    def removeUndirEdgesFromNode(self, V):
        edgesToRemove = set()
        for edgeSet in self.undirEdges:
            if V in edgeSet:
                edgesToRemove.add(edgeSet)
        self.undirEdges = self.undirEdges - edgesToRemove
    
    def toDot(self, outpath):
        text = "digraph {\n"
        
        # Sort levels
        levels = {}
        for node in self.nodes:
            level = self.nodeLevel[node]
            levels[level] = levels.get(level, []) + [node]
        
        # Add nodes
        for node in self.nodes:
            text += f"{node}; "
        text += "\n"
        
        # Add undirected edges
        text += "subgraph Undirected {\n"
        text += "edge [dir=none, color=red]\n"
        for edgeSet in self.undirEdges:
            edgeSet = list(edgeSet)
            text += f"{edgeSet[0]} -> {edgeSet[1]}\n"
        
#         for level, nodes in levels.items():
#             text += f"{{rank = same; {';'.join(nodes)}; }}\n"
        text += "}\n\n"
        
        # Add directed Edges
        text += "subgraph Directed {\n"
        text += "edge [color=black]\n"
        for edgeSet in self.dirEdges:
            edgeSet = list(edgeSet)
            text += f"{edgeSet[0]} -> {edgeSet[1]}\n"
        
#         for level, nodes in levels.items():
#             text += f"{{rank = same; {';'.join(nodes)}; }}\n"
        text += "}\n\n"
        
        text += "}\n"
        
        with open(outpath, "w") as f:
            f.write(text)
