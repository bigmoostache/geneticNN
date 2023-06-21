import warnings

#simple class representing a graph of the networks
#the graph must be unidiretional (there can only be a
#path a->b or b->a not not both)

class Graph():
    #nodes = list of existing nodes
    #edges = list of edges between nodes
    #   edges are unidirectional and are backward w.r.t 
    #  calculation order
    def __init__(self,nodes,runs,outputs, inputs =0):
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
                if run['inputs'][input][0] == 0:
                    inputs.append(run['inputs'][input][1])
        return inputs
    def check_graph(self):
        for e in self.runs:
            #check nodes exist
            if not e['id'] in self.nodes:
                return False
            for i in list(e['inputs']):
                if not e['inputs'][i][0] in self.nodes and e['inputs'][i][0] != 0:
                    s = "when checking graph, the following node was referred in the edges but not existing : ", + e['inputs'][i][0]
                    warnings.warn(s, category = Warning)
                    return False
        return True
            #we don't check for unidirectionality
            
    def get_child_runs(self,node,with_indices = False):
        child_runs = []
        if not with_indices:
            for e in self.runs:
                if e['id'] is node:
                    child_runs.append(e)
            return child_runs
        for index,e in enumerate(self.runs):
            if e['id'] is node:
                child_runs.append([index,e])
        return child_runs

    def check_parent_child_exist(self,parent,child):
        if not parent in self.nodes and not parent != 0:
            raise Exception("unknown parent node: ", parent)
        if not child in self.nodes:
            raise Exception("unknown child node: ", child)
        
    def is_parent(self,parent,child,stop_at_loop = True):
        self.check_parent_child_exist(parent,child)
        return self._is_parent_no_err(parent,child,stop_at_loop)
    
    def _is_parent_no_err(self,parent,child,stop_at_loop = True):
        child_runs = self.get_child_runs(child)
        
        # check direct connection:
        for e in child_runs:
            for input in e['inputs']:
                if e['inputs'][input][0] is parent:
                    return True

        #no direct connection, search in parents of child
        for e in child_runs:
            for input in e['inputs']:
                if self._is_parent_no_err(parent,e['inputs'][input][0]):
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
        for e in self.get_child_runs(child):
            for input in e['inputs']:
                if e['inputs'][input][0] == parent:
                    return True
        return False

    def add_run(self,child,parent_dict):
        for key in parent_dict:
            self.check_parent_child_exist(parent_dict[key][0],child)
        run = {'id': child, 'inputs': parent_dict}
        self.runs.append(run)


    def get_children(self,node):
        next_nodes = []
        for e in self.runs:
            for input in list(e['inputs']):
                if e['inputs'][input][0] == node:
                    next_nodes.append(e['id'])
        return next_nodes
    
    def get_children_run(self,node):
        next_nodes = []
        for e in self.runs:
            for input in list(e['inputs']):
                if e['inputs'][input][0] == node:
                    next_nodes.append(e)
        return next_nodes
    
    def get_children_runs_wt_indices(self,node):
        next_nodes = []
        for i,e in enumerate(self.runs):
            for input in list(e['inputs']):
                if e['inputs'][input][0] == node:
                    next_nodes.append([i,e])
        return next_nodes

    def get_parents(self,node):
        parents = []
        for e in self.runs:
            if e["id"] == node:
                for input in list(e['inputs']):
                    parents.append(e['inputs'][input][0])
        return parents
    def get_parents_runs(self,node):
        parents = []
        for e in self.runs:
            if e["id"] == node:
                parents.append(e)
        return parents
    
    def get_parents_runs_wt_indices(self,node):
        parents = []
        for i,e in enumerate(self.runs):
            if e["id"] == node:
                parents.append([i,e])
        return parents