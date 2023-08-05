import warnings
import numpy as np

'''
A class that contains the barebone structure describing a model: the input and output of the model, 
the different layers used, the order in which variables are passed in layers and
the parameters of the model and their transcription to parameters of its layers

The inputs of a run are give as its
'''



class Model_Skeleton():
    def __init__(self,inputs, outputs, layers, runs,parameters_list = ''):
        self.layers = layers
        self.runs = runs
        self.outputs = outputs
        self.inputs = inputs
        if self.inputs == 0:
            self.inputs = self.find_inputs()


    def find_inputs(self):
        inputs = []
        for run in self.runs:
            for input in run['inputs']:
                if run['inputs'][input][0] == -1:
                    inputs.append(run['inputs'][input][1])
        return inputs

        if not self.check_graph():
            raise Exception("The graph is not well constructed: it has edges refering to non-existing nodes")
        
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
    
    def is_parent(self,parent_id,child_id):
         
        if child_id is -1:
            return parent_id is -1

        direct_parents = self.get_direct_parents(child_id)
        # check direct connection:
        if parent_id in direct_parents:
            return True

        #no direct connection, search in parents of child
        for i in direct_parents:
            if self.is_parent(parent_id,i):
                return True
                
        return False
    
    # returns the runs whose outputs are input of the given run
    def get_direct_parents(self,run_id):
        if run_id ==-1:
            return[]
        parents = []
        for input in self.runs[run_id]['inputs']:
            parents.append(self.runs[run_id]['inputs'][input][0])
        return list(set(parents))
    

    def get_parents(self,run_id):
        if run_id ==-1:
            return[]
        parents = []
        direct_parents = self.get_direct_parents(run_id)
        parents.extend(direct_parents)
        run = self.runs[run_id]
        for parent in direct_parents:
            parents.extend(self.get_parents(parent))
        return parents
    
    def get_direct_children(self,run):
        child_runs = []
        for index,run in enumerate(self.runs):
            for input in run['inputs']:
                if run['inputs'][input][0] == run:
                    child_runs.append(index)
        return child_runs

    def get_children(self,run):
        children = []
        direct_children = self.get_direct_children(run)
        children.extend(direct_children)
        for child in direct_children:
            children.extend(self.get_children(child))
        return list(set(children))
    

    def is_connected_to_output(self,run):
        outputing_runs=  []
        for key in self.outputs:
            outputing_runs.append(self.outputs[key]['run_id'])
        outputing_run = set(outputing_runs)
        for run_ in outputing_runs:
            if run_ == run or self.is_parent(run,run_):
                return True
        return False
    
    def is_connected_to_input(self,run):
        return self.is_parent(-1,run)

    def get_runs_heights(self):
        runs_size = len(self.runs)
        heights = [-1 for i in range(runs_size)]

        def set_height(run_id,height):
            if run_id is -1:
                return
            if height > runs_size:
                raise Exception('Cycle detected in the graph')
            if heights > heights[run_id]:
                heights[run_id] = height
            
                for input in self.runs[run_id]['inputs']:
                    set_height(self.runs[run_id]['inputs'][input][0], height + 1)
            
        for output in self.outputs:
            set_height(self.outputs[output]["run_id"], 0)
        return heights
    
    def get_unused_runs(self):
        unused_runs = []
        heights = self.get_runs_heights()
        for i,h in enumerate(heights):
            if h is -1:
                unused_runs.append(i)

        return unused_runs
    
    def find_runs_order(self):
        heights = self.get_runs_heights()

        runs_size = len(self.runs)
        max_height = max(heights)
        ordered_items = []
        for i in range(max_height+1):
            for j in range(runs_size):
                if heights[j] == i:
                    ordered_items.append(j)

        return ordered_items
    
    def reorder_runs(self):
        new_order = self.find_runs_order()
        inverser = [-1 for i in range(len(self.runs))]
        for i in new_order:
            inverser[i] = new_order.index(i)
        #print('obtained new runs order: ', new_order)
        #print('inverser: ', inverser)
        self.runs = [self.runs[i] for i in new_order]
        for run in self.runs:
            for input in run['inputs']:
                if run['inputs'][input][0] != -1:
                    run['inputs'][input][0] = inverser[run['inputs'][input][0]]
        for output in self.outputs:
            self.outputs[output]['run_id'] = inverser[self.outputs[output]['run_id']]

        
    def get_runs_of_model(self,model_id):
        node_runs = []
        for i,run in enumerate(self.runs):
            if run['id'] is model_id:
                node_runs.append(i)
        return node_runs
    
    def get_first_run_of_model(self,model_id):
        for i,run in enumerate(self.runs):
            if run['id'] is model_id:
                return i
        return None
    
    def del_run(self,index):
        for run in self.runs:
            for input in run['inputs']:
                if run['inputs'][input][0] > index:
                    run['inputs'][input][0] -= 1
        for output in self.outputs:
            if self.outputs[output]['run_id'] > index:
                self.outputs[output]['run_id'] -= 1   
        del self.runs[index]

    def add_model(self,model_parameters):
        keys = list(self.nodes)
        node_type = model_parameters['template']
        nums = [s.split(node_type + '_')[1] for s in keys if node_type in s]
        nums = sorted([int(s) for s in nums and s.isdigit()])
        n_used = -1
        for i in range(len(nums) - 1):
            if nums[i+1] - nums[i] != 1:
                    n_used = nums[i] +1
        if n_used == -1:
            n_used = nums[-1] + 1
        model_name = node_type + '_'  + n_used
        self.nodes[model_name] = model_parameters
        return model_parameters
    
    def add_run(self,run_to_insert ):
        self.append(run_to_insert)
        return len(self.runs) - 1