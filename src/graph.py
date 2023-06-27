import warnings


#simple class representing a graph of the networks
#the graph must be unidiretional (there can only be a
#path a->b or b->a not not both)

class Graph():
    #nodes = list of existing nodes
    #edges = list of edges between nodes
    #   edges are unidirectional and are backward w.r.t 
    #  calculation order
    def __init__(self,nodes,runs,outputs,inputs =0):
        self.nodes = nodes
        self.runs = runs
        self.outputs = outputs
        self.inputs = inputs
        if self.inputs == 0:
            self.inputs = self.find_inputs()

        if not self.check_graph():
            raise Exception("The graph is not well constructed: it has edges refering to non-existing nodes")
        
    def find_inputs(self):
        inputs = []
        for run in self.runs:
            for input in run['inputs']:
                if run['inputs'][input][0] == -1:
                    inputs.append(run['inputs'][input][1])
        return inputs
    def check_graph(self):
        for e in self.runs:
            #check nodes exist
            if not e['id'] in self.nodes:
                return False
            for i in list(e['inputs']):
                if (not e['inputs'][i][0] <= len(self.runs)) or not e['inputs'][i][0] >= -1:
                    s = "when checking graph, the following run was referred in the edges inputs but not existing : ", + e['inputs'][i][0]
                    warnings.warn(s, category = Warning)
                    return False
        return True
            #we don't check for unidirectionality
            
    

    def check_parent_child_exist(self,parent,child):
        if not parent in self.nodes and not parent != 0:
            raise Exception("unknown parent node: ", parent)
        if not child in self.nodes:
            raise Exception("unknown child node: ", child)
        
        

    def is_parent(self,parent,child,stop_at_loop = True):
        self.check_parent_child_exist(parent,child)
        return self._is_parent_no_err(parent,child,stop_at_loop)
    


    def _is_parent_no_err(self,parent,child,stop_at_loop = True):
        child_runs = self.get_children(child)
        
        # check direct connection:
        for i in child_runs:
            run = self.runs[i]
            for input in run['inputs']:
                if run['inputs'][input][0] is parent:
                    return True

        #no direct connection, search in parents of child
        for i in child_runs:
            run = self.runs[i]
            for input in run['inputs']:
                if self._is_parent_no_err(parent,run['inputs'][input][0]):
                    return True
                
        return False
    

    def is_direct_parent(self,parent,run):
            for input in run['inputs']:
                if run['inputs'][input][0] is parent:
                    return True
    def is_connected_to_output(self,node):
        outputing_nodes=  []
        for key in self.outputs:
            outputing_nodes.append(self.outputs[key]['model_id'])
        outputing_nodes = set(outputing_nodes)
        for node_ in outputing_nodes:
            if node_ == node or self.is_parent(node,node_):
                return True
        return False
        
    def exists_edge(self,parent,child):
        for i in self.get_parents(child):
            if parent is i:
                return True
        return False

    def add_run(self,run_model_id,parent_dict):
        for key in parent_dict:
            self.check_parent_child_exist(self.runs[parent_dict[key][0]]['id'],run_model_id)
        run = {'id': run_model_id, 'inputs': parent_dict}
        self.runs.append(run)

    def find_runs_order(self):
        #first sort runs:
        items = [[run['inputs'][input][0] for input in run['inputs']] for run in self.runs]
        def compare_func(item1,item2):
            if item2 in items[item1]:
                return True
            return False
        
        def topological_sort_with_comparison(items, compare):
            ordered_items = []

            remaining_items = list(range(len(items)))

            def compare_items(item1, item2):
                if item1 == item2:
                    return 0
                elif compare(item1, item2):
                    return -1
                else:
                    return 1

            def compare_and_sort(current_item, remaining_items):
                group1, group2, group3 = [], [], []
                ordered_items_ = []
                for item in remaining_items:
                    comparison = compare_items(current_item, item)
                    if comparison == -1:
                        group1.append(item)
                    elif comparison == 0:
                        group2.append(item)
                    else:
                        group3.append(item)

                if group1:
                    ordered_items_.extend(compare_and_sort(group1[0], group1[1:]))

                ordered_items_.append(current_item)

                if len(group3) > 1:
                    ordered_items_.extend(compare_and_sort(group3[0], group3[1:]))
                
                if len(group3) == 1:
                    ordered_items_.extend(group3)
                return ordered_items_
            ordered_items = compare_and_sort(0, remaining_items)
            print(ordered_items)
            return [items[i] for i in ordered_items]

        return topological_sort_with_comparison(items,compare_func)

    def reorder_runs(self):
        new_order = self.find_runs_order()
        print('obtained new runs order: ', new_order)
        self.runs = [self.runs[i] for i in new_order]
    
    def get_children(self,node):
        child_runs = []
        for index,e in enumerate(self.runs):
            if e['id'] is node:
                child_runs.append(index)
        return child_runs
    
    def get_parents(self,run_id):
        parents = []
        run = self.runs[run_id]
        for input in list(run['inputs']):
            parents.append(run['inputs'][input][0])
        return parents
    

    def get_runs_of_model(self,node_id):
        node_runs = []
        for i,run in enumerate(self.runs):
            if run['id'] is node_id:
                node_runs.append(i)
        return node_runs