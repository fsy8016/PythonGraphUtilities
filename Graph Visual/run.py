#Forest Yan, yanxx232

from tree import Tree
import time

(ROOT, DEPTH, BREADTH) = range(3)


def main():
    tree = Tree()
    
    #set root node
    root = "Forest"
    
    mode = input('ENTER 0 FOR DFS, ENTER 1 FOR BFS\nNOTE: Strings must be between quotes\n')
    #print(mode)
    #print(mode == 1)
    
                
    #add elements to the graph
    #USAGE: tree.add_node(leaf,parent)
    
    tree.add_node(root)
    tree.add_node("Steve", root)
    tree.add_node("Kao", root)
    tree.add_node("Diplo",root)
    tree.add_node("Lol", "Steve")
    tree.add_node("Amy", "Steve")
    tree.add_node("Julio", "Amy")
    tree.add_node("Mary", "Amy")
    tree.add_node("Mark", "Julio")
    tree.add_node("Jane", "Mark")
    tree.add_node("Tahir", "Kao")
    tree.add_node("What", "Tahir")
    
    tree.display(root)
    
    if(mode == 0):
        print("DEPTH-FIRST ITERATION:\n")
        for node in tree.traverse(root):
            tree.specialdisplay(root,node)
            print("\n")
            time.sleep(1)
    if(mode == 1):
        print("BREADTH-FIRST ITERATION:\n")
        for node in tree.traverse(root, mode=BREADTH):
            tree.specialdisplay(root,node)
            print("\n")
            time.sleep(1)
    
    restart = input('RESTART? ("y" or "n")\n')
    if(restart == 'y' or restart == 'Y'):
        main()
            
main()