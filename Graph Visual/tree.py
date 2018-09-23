#Forest Yan, yanxx232

from node import Node

#bold for the terminal if usable
class color:
   PURPLE = '\033[95m'
   CYAN = '\033[96m'
   DARKCYAN = '\033[36m'
   BLUE = '\033[94m'
   GREEN = '\033[92m'
   YELLOW = '\033[93m'
   RED = '\033[91m'
   BOLD = '\033[1m'
   UNDERLINE = '\033[4m'
   END = '\033[0m'

(ROOT, DEPTH, BREADTH) = range(3)


class Tree:
    #initialize empty array for tree
    def __init__(self):
        self.__nodes = {}

    @property
    def nodes(self):
        return self.__nodes
        
    #intakes a parent and child and adds it to the tree 
    
    def add_node(self, ID, parent=None):
        node = Node(ID)
        self[ID] = node

        if parent is not None:
            self[parent].add_child(ID)

        return node
        
    #display for each depth of the tree, used only mainly for the initial show of the tree
    
    def display(self, ID, depth=ROOT):
        children = self[ID].children
        if depth == ROOT:
            print("{0}".format(ID))
        else:
            print('\t'*depth),
            print("{0}".format(ID))

        depth += 1
        for child in children:
            self.display(child, depth)
    
    #iterate through tree and mark the current node.
    
    def specialdisplay(self, ID, node, depth=ROOT):
        children = self[ID].children
        if depth == ROOT:
            if("{0}".format(ID) == node):
                print("CURRENT:\n" + '\t'*depth + color.BOLD + "{0}".format(ID) + color.END)
            else:
                print("{0}".format(ID))
        else:
            print('\t'*depth),
            if("{0}".format(ID) == node):
                print("CURRENT:\n" + '\t'*depth + color.BOLD + "{0}".format(ID) + color.END)
            else:
                print("{0}".format(ID))
        depth += 1
        for child in children:
            self.specialdisplay(child, node, depth)
        
            
    #dynamically goes through each root node in queue
    #referenced and changed from:
    #http://www.quesucede.com/page/show/id/python-3-tree-implementation
    
    def traverse(self, ID, mode=DEPTH):
        yield ID
        queue = self[ID].children
        while queue:
            yield queue[0]
            exp = self[queue[0]].children
            if mode == DEPTH:
                #DFS
                queue = exp + queue[1:]
            elif mode == BREADTH:
                #BFS
                queue = queue[1:] + exp

    #our basic get/set
    def __getitem__(self, key):
        return self.__nodes[key]

    def __setitem__(self, key, item):
        self.__nodes[key] = item