import numpy as np
import random 
import pandas as pd
from itertools import combinations,permutations 
import nltk
import sys
from modules import visualize_sol as vis_sol
from modules import random_instance as rand_inst 
from modules import utility as util 
import time
from matplotlib import pyplot as plt
import copy


class InitialRouteGenerator:
    def __init__(self,_no_layer, _labeling_dict, customer_demand, constant_dict, distance_matrix):
        # CONSTANTs
        self.no_truck = _no_layer
        self.no_dock = len(_labeling_dict['docking'])
        self.no_customer = len(_labeling_dict['customers'])
        self.customer_demand = customer_demand
        self.distance_matrix = distance_matrix
        self.truck_capacity = constant_dict['truck_capacity']
        self.fixed_setup_time = constant_dict['fixed_setup_time']
        self.truck_speed = constant_dict['truck_speed']        
        
        self.depot = _labeling_dict['depot']
        self.depot_s = _labeling_dict['depot_s']
        self.depot_t = _labeling_dict['depot_t']
        self.all_depot = _labeling_dict['all_depot']
        self.nodes = _labeling_dict['nodes']
        self.customers = _labeling_dict['customers']
        self.arcs = _labeling_dict['arcs']
    def generateArcs(self):
        #Not nesessary as arcs already generated with nodeSet
        nodes = set()
        nodes = list(nodes.union(set(self.docking),set(self.customers),set(self.depot)))
        no_node = len(nodes)
        node_combi = list(combinations(nodes,2))
        arc_permute = [list(permutations(list(c))) for c in node_combi ]
        sh = np.shape(arc_permute)
        arc_permute = np.reshape(arc_permute,(sh[0]*sh[1],sh[2])).tolist()
        arc_permute = [','.join(list(l)) for l in arc_permute]
#         print(arc_permute)
        arc_permute = self.splitDepotArcsVar(arc_permute,self.depot,self.depot_s,self.depot_t)
        print("Finished creating truck_arcs:",len(arc_permute))
        return arc_permute
    
    def splitDepotArcsVar(self,a_var,depot,depot_s,depot_t):
        '''INPUT: ['depot,customer_1,T','customer_2,depot,T']
        OUTPUT:['depot_s,customer_1,T','customer_2,depot_t,T'] '''
        new_a_var =[]
        for a in a_var:
            v = a.split(',')
            if len(v)>1:
                if v[0]==depot[0]: v[0]=depot_s[0]
                if v[1]==depot[0]: v[1]=depot_t[0]
            new_a_var.append(','.join(v))
        return new_a_var
    
    def mergeDepotArcsVar(self,a_var,depot,depot_s,depot_t):
        '''INPUT: ['depot_s,customer_1,T','customer_2,depot_t,T']
        OUTPUT: ['depot,customer_1,T','customer_2,depot,T']'''
        new_a_var =[]
        for a in a_var:
            v = a.split(',')
            if len(v)>1:
                if v[0]==depot_s[0] or v[0]==depot_t[0]:v[0]=depot[0]
                if v[1]==depot_s[0] or v[1]==depot_t[0]:v[1]=depot[0]
            else:
                if v[0]==depot_s[0] or v[0]==depot_t[0]:v[0] = depot[0]
            new_a_var.append(','.join(v))
        return new_a_var    
    
    def generateAllCombiNodes(self,max_visited_nodes=None,str_node=None,end_node=None):
        '''THIS FUNCTION WILL GENERATE ALL POSSIBLE COMBINATIONS OF INPUT NODE LIST'''
        all_combi_nodes = []
        for i in range(1,max_visited_nodes+1):
            combination_set = list(combinations(self.customers,i))
            permutation_set = [list(permutations(list(c))) for c in combination_set ]
        #     print(permutation_set,np.shape(permutation_set))
            sh = np.shape(permutation_set)
        #     print(np.reshape(permutation_set,(sh[0]*sh[1],sh[2])))
            re_perm_set = np.reshape(permutation_set,(sh[0]*sh[1],sh[2])).tolist()
            all_combi_nodes = all_combi_nodes+re_perm_set
        # ADD depot 
        if (str_node is not None) and (end_node is not None):
            all_combi_nodes = [str_node+p+end_node for p in all_combi_nodes]
        else:all_combi_nodes = [p for p in all_combi_nodes]
        return all_combi_nodes
    
    def generateSetOfArcsFromNodesCombi(self,all_combi_nodes,added_type=''):
        '''THIS FUNCTION WILL GENERATE ARCS LIST BY CREATING BIGRAMS OF INPUT COMBI NODES''' 
        node2arcs = []
        for r in all_combi_nodes:    
            if len(r)==1: arc_list = [r]
            else: arc_list = list(nltk.bigrams(r))
            if added_type!='':arc_list = [','.join(list(a)+[added_type]) for a in arc_list]
            else:arc_list = [tuple(a) for a in arc_list]
            node2arcs.append(arc_list)
        return node2arcs
    
    def generateRoutes(self,initRouteDf,truck_cap_limit=None,
                       max_visited_nodes=None, max_vehicles_per_route=None, 
                       clustered=None, nbInitRoute=None,
                       mode=None,drone_cap_limit=None):
        if max_visited_nodes is None: max_visited_nodes=len(self.customers)
        if max_vehicles_per_route is None: max_vehicles_per_route=len(self.customers)
        if truck_cap_limit is None: truck_cap_limit = self.truck_capacity
        self.all_combi_nodes=list()
        self.routes_arcs=list()
        t1 = time.time()
        
        _all_combi_nodes = self.generateAllCombiNodes(max_visited_nodes,self.depot,self.depot)
        self.all_combi_nodes+= _all_combi_nodes
        self.routes_arcs += self.generateSetOfArcsFromNodesCombi(_all_combi_nodes,'')
        
        if nbInitRoute is None: nbInitRoute = len(self.all_combi_nodes)
        print('nbInitRoute is set to (#UniqueSequences) * (#MaxVehiclesPerRoute) = {}*{} = {}'.format(nbInitRoute,max_vehicles_per_route,nbInitRoute*max_vehicles_per_route))
        
        previous_cols = initRouteDf.columns[initRouteDf.columns.str.contains('r')].shape[0]
        ## ADD NEW COL TO DATAFRAME
        counter = 0
        for idx in range(len(self.all_combi_nodes)):
            if ((idx/len(self.all_combi_nodes))*100 % 10)==0: 
                print('progress:',idx*max_vehicles_per_route,'/',len(self.all_combi_nodes)*max_vehicles_per_route)
#             if idx>nbInitRoute: break
            lr_route = self.calculateLr(self.routes_arcs[idx])
            qr_route = pd.Series(self.all_combi_nodes[idx]).apply(lambda x: self.customer_demand[x]).sum()
            veh_min = int(np.ceil(qr_route*lr_route/truck_cap_limit))
#             print(veh_min)
            if (veh_min > max_vehicles_per_route): continue
            else:
                for veh_no in range(veh_min,max_vehicles_per_route+1):
                    if (veh_no < veh_min)  and (mode!='all'): continue
                    else:
                        self.addNewCol(initRouteDf, lr_route, veh_no,self.all_combi_nodes[idx],self.routes_arcs[idx],'route['+str(counter)+']')
                        counter += 1; #print(veh_no,veh_min)
#         initRouteDf['labels']=init_path.splitDepotArcsVar(initRouteDf.labels,self.depot,[self.all_depot[0]],[self.all_depot[1]])
        self.init_routes_df = initRouteDf.copy()
#         self.init_routes_df = self.init_routes_df.set_index('labels')
        print('#Feasible Cols:', len(initRouteDf.columns)-1)
        print('Elapsed-time:',time.time()-t1)
    
    def generateBasicInitialPatterns(self,nbInitRoute,initRouteDf):
        columns = ['PathCoeff']
        path = pd.DataFrame(index = range(nbInitRoute), columns = columns)
        path['PathCoeff']=[initRouteDf[initRouteDf.columns[idx]].values for idx in range(nbInitRoute)]
        return path
    
    
    def addNewCol(self, df, col_cost, veh_no, nodes, arcs, var_name=None):
        coeff = nodes+arcs
        if var_name is None:var_name = df.columns.shape[0]
        column = df.labels.isin(coeff).astype(int)
        df[var_name]=column
        df.loc[df['labels']=='m',var_name] = veh_no
        df.loc[df['labels']=='lr',var_name] = col_cost
    
    def calculateLr(self, route_arcs):
        lr = self.fixed_setup_time+(pd.Series(route_arcs).apply(lambda x:self.distance_matrix[x]).sum()/self.truck_speed)
        return lr
   
    def generateInitDF(self, _row_labels,_constant_dict):
        n = len(self.customers)
        init_max_npr = _constant_dict['init_max_nodes_proute']
        init_max_mpr = _constant_dict['init_max_vehicles_proute']
        U = [[0]]
        P = []
        init_route_df = pd.DataFrame(data =_row_labels,columns=['labels'])
        counter = 0
        terminate = False
        t1 = time.time()
        while not terminate:
            cS = U.pop(0)
            for j in range(1,n+1):
                if (j not in cS) and (len(cS)-1) < init_max_npr:
                    nS = cS + [j]
                    U.append(nS)
                    nS_lr = 0
                    coeff = []
                    for idx in range(len(nS)):
                        if idx==0: i='O'
                        else: i = 'c_%s'%(nS[idx])
                        if idx==(len(nS)-1): j = 'O'
                        else: j = 'c_%s'%(nS[idx+1])
                        coeff += [i,(i,j)]
                        nS_lr+=self.distance_matrix[(i,j)]/self.truck_speed
        #                     print(coeff)
                    nS_lr+=self.fixed_setup_time
                    nS_qr = 0
                    for c_i in range(1,len(nS)): nS_qr+=self.customer_demand['c_%s'%(nS[c_i])]
                    veh_min = int(np.ceil(nS_qr*nS_lr/self.truck_capacity))
                    if (veh_min > init_max_mpr): continue # not feasible even for p veh
                    else:
                        column = init_route_df.labels.isin(coeff).astype(int)
                        column.iloc[0] = nS_lr
                        for veh_no in range(veh_min,init_max_mpr+1):
                            if ((np.log10(counter+1)) % 0.5)==0: 
                                print('processed route:',counter+1)
                            var_name = 'route['+str(counter)+']'
                            column.iloc[1] = veh_no
                            init_route_df[var_name]=column.values
                            counter+=1
                        
            if len(U)==0: terminate=True
        self.initColsTe = time.time()-t1
        print('Total: %d routes'%(counter-1))
        print('Elapsed Time:',self.initColsTe)
        self.init_routes_df = init_route_df
        init_route = self.generateBasicInitialPatterns(init_route_df.shape[1]-1,initRouteDf=init_route_df.set_index('labels'))
        init_route.rename(index=lambda x:'route[%d]'%x,inplace=True)
        return init_route