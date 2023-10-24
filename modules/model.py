import sys
from modules import visualize_sol as vis_sol
from modules import initialize_path as init_path
from modules import random_instance as rand_inst 
from modules import utility as util 
# import branch_and_price as bnp
import pandas as pd
import time
import numpy as np
from gurobipy import *
from operator import itemgetter
import os
os.environ['GRB_LICENSE_FILE'] = '/Users/ravitpichayavet/gurobi.lic'
epsilon = 1e-5

class TRPModel:
    def __init__(self, _init_route, _initializer,
                 _distance_matrix,constant_dict,
                 extra_constr=None, _model_name = "PhaseII", _mode=None, relax_route=False):
        
        self.init_route = _init_route.copy()
        self.route_coeff = _init_route['PathCoeff'].values
        self.init_routes_df = _initializer.init_routes_df.copy()
        self.relax_route_flag = relax_route
        
        self.depot = _initializer.depot
        self.depot_s = _initializer.depot_s
        self.depot_t = _initializer.depot_t
        self.all_depot = _initializer.all_depot
        self.customers = _initializer.customers
        self.nodes = _initializer.nodes
        self.arcs = _initializer.arcs
        self.route_cost = []
        
        self.coeff_series = _initializer.init_routes_df['labels']
        self.depot_index = self.coeff_series[self.coeff_series.isin(self.all_depot)].index.values
        self.depot_s_index = self.coeff_series[self.coeff_series.isin(self.depot_s)].index.values
        self.depot_t_index = self.coeff_series[self.coeff_series.isin(self.depot_t)].index.values
        self.customer_index = self.coeff_series[self.coeff_series.isin(self.customers)].index.values
        self.nodes_index = self.coeff_series[self.coeff_series.isin(self.nodes)].index.values
        self.arcs_index = self.coeff_series[self.coeff_series.isin(self.arcs)].index.values
        self.veh_no_index = self.coeff_series.loc[self.coeff_series=='m'].index.values
        self.lr_index = self.coeff_series.loc[self.coeff_series=='lr'].index.values
    
        self.constant_dict = constant_dict.copy()
        self.vehicle_capacity = self.constant_dict['truck_capacity']
        self.fixed_setup_time = self.constant_dict['fixed_setup_time']
        self.truck_speed = self.constant_dict['truck_speed']
        self.distance_matrix = _distance_matrix
        self.customer_demand = _initializer.customer_demand
        self.max_vehicles = self.constant_dict['max_vehicles']
        self.max_nodes_proute_DP = self.constant_dict['max_nodes_proute_DP']
        self.max_vehicles_proute_DP = self.constant_dict['max_vehicles_proute_DP']
        
        self.route_index = pd.Series(self.init_route.index).index.values
        self.model = Model(_model_name)
        if _mode is None: self.mode='multiObjective'
        else: self.mode=_mode
        
        self.cost_matrix = dict()
        for k,v in self.distance_matrix.items():
            if k[0].split('_')[-1] == "O":
                i = 0
            else: 
                i = int(k[0].split('_')[-1])
            if k[1].split('_')[-1] == "O":
                j = 0
            else:
                j = int(k[1].split('_')[-1])
            nk = (i,j)
            self.cost_matrix[nk] = v/self.truck_speed
        
        self.DPRouteDict=dict()
            
    def buildModel(self):
        self.generateVariables()
        self.generateConstraints()
        self.generateCostOfRoutes()
        self.generateObjective()
        self.model.update()
        
    def generateVariables(self):
        self.route = self.model.addVars(self.route_index, lb=0,
                                       vtype=GRB.BINARY, name='route')
        print('Finish generating variables!')
        
    def generateConstraints(self):  
        if self.relax_route_flag:
            const1 = ( quicksum(self.route_coeff[rt][i]*self.route[rt] for rt in self.route_index) >= int(1) \
                                 for i in self.customer_index )
            self.model.addConstrs( const1,name='customer_coverage' )
            const2 = (quicksum(self.route[rt]*(self.route_coeff[rt][self.veh_no_index[0]]) for rt in self.route_index)<=self.max_vehicles)
            self.model.addConstr(const2,name='vehicles_usage' )
        else: 
            const1 = ( quicksum(self.route_coeff[rt][i]*self.route[rt] for rt in self.route_index) == int(1) \
                                 for i in self.customer_index )
            self.model.addConstrs( const1,name='customer_coverage' )
            const2 = (quicksum(self.route[rt]*(self.route_coeff[rt][self.veh_no_index[0]]) for rt in self.route_index)==self.max_vehicles)
            self.model.addConstr(const2,name='vehicles_usage' )
        print('Finish generating constrains!')
    
    def generateCostOfRoutes(self):
        t1=time.time()
        self.route_cost = self.init_routes_df.set_index('labels').apply(lambda col: self.calculateCostOfRoute(col),axis=0)
        print('Finish generating cost vector!....Elapsed-time:',time.time()-t1)

    def calculateCostOfRoute(self, route):
        ''' Calculate 3 costs for each route: 
            1) '''
        visiting_nodes = pd.Series(route[self.customer_index][route>=1].index)
        visiting_arcs = pd.Series(route[self.arcs_index][route>=1].index)
        next_node = ['STR']
        dem_weighted_distance = 0
        qr = visiting_nodes.apply(lambda x: self.customer_demand[x]).sum()
        avg_waiting = qr*route['lr']/(2*route['m'])
        visited_node = []
        demand_travel_time_dict = dict(zip(visiting_nodes,[route['lr']*self.constant_dict['tw_avg_factor']/route['m']]*len(visiting_nodes)))
        acc_distance = 0
        while next_node[0]!=self.depot[0]:
            if next_node[0] == 'STR': next_node.pop(0);selecting_node = self.depot[0] #only for the first arc
            else: selecting_node = next_node.pop(0)
#             print(selecting_node)
#             print(visited_node)
            outgoing_arc_list = visiting_arcs[visiting_arcs.apply(lambda x: ((x[0]==selecting_node) and (x[1] not in visited_node) ))].to_list()
            if (selecting_node != self.depot[0]): visited_node.append(selecting_node)
#             print("outgoing",outgoing_arc_list)
            outgoing_arc = outgoing_arc_list[0]
            node_j = outgoing_arc[1]
            next_node.append(node_j)
            qj = self.customer_demand[node_j]
            traveling_time_carrying_pkg = qr*(self.distance_matrix[outgoing_arc])/self.constant_dict['truck_speed']
            acc_distance +=(self.distance_matrix[outgoing_arc])/self.constant_dict['truck_speed']
            dem_weighted_distance+=traveling_time_carrying_pkg
            # remove delivered demand
            qr = qr-qj
            if node_j!=self.depot[0]:
                demand_travel_time_dict[node_j] += acc_distance
        cost_dict = dict(zip(['total_cost','avg_waiting','dem_weighted_distance','dem_waiting'],[dem_weighted_distance+avg_waiting,avg_waiting,dem_weighted_distance,demand_travel_time_dict]))
        return cost_dict
        
    def generateObjective(self):
        # Minimize the total cost of the used rolls
        if self.mode=='multiObjective':
            self.model.setObjective( quicksum(self.route[rt]*(self.route_cost[rt]['total_cost']) for rt in self.route_index) ,
                                    sense=GRB.MINIMIZE)
        elif self.mode=='TSPOnly':
            self.model.setObjective( quicksum(self.route[rt]*(self.route_cost[rt]['avg_waiting']) for rt in self.route_index) ,
                                    sense=GRB.MINIMIZE)
        elif self.mode=='TRPOnly':
            self.model.setObjective( quicksum(self.route[rt]*(self.route_cost[rt]['dem_weighted_distance']) for rt in self.route_index) ,
                                    sense=GRB.MINIMIZE)
        print('Finish generating objective!')
        
    def solveModel(self, timeLimit = None,GAP=None):
        if timeLimit is not None: self.model.setParam('TImeLimit', timeLimit)
        if GAP is not None: self.model.setParam('MIPGap',GAP)
        self.model.setParam('SolutionNumber',2)
        self.model.setParam(GRB.Param.PoolSearchMode, 2)
        self.model.optimize()
        
    ##RELAXATION
    def solveRelaxedModel(self):
        #Relax integer variables to continous variables
        self.relaxedModel = self.model.relax()
        var_ss = pd.Series(self.relaxedModel.getVars())
        var_ss.apply(lambda x: x.setAttr('ub',GRB.INFINITY))
        self.relaxedModel.optimize()
        
    def solveRelaxedBoundedModel(self):
        #Relax integer variables to continous variables <=1
        self.relaxedBoundedModel = self.model.relax()
        var_ss = pd.Series(self.relaxedBoundedModel.getVars())
        var_ss.apply(lambda x: x.setAttr('ub',1))
        self.relaxedBoundedModel.optimize()
        
    def getRelaxSolution(self):
        a = pd.Series(self.relaxedModel.getAttr('X'))
        return a[a>0]

    def getDuals(self):
        return self.relaxedModel.getAttr('Pi',self.model.getConstrs())
    
    ##COLUMNS GENERATION
    def addColumn(self,_col_object,_col_cost,_name):
        self.model.addVar(lb=0,vtype=GRB.BINARY,column=_col_object, obj= _col_cost, name=_name)
        self.model.update()
    
    def generateColumns(self,_filtered_df,_duals, ):
        for index, row in _filtered_df.iterrows():
            _col = row.colDF.loc[row.colDF.index[self.customer_index]].iloc[:,-1].to_list() +row.colDF.loc[row.colDF.labels=='m'].iloc[:,-1].to_list()
            newColumn = Column(_col, self.model.getConstrs())
            _name = row.colDF.columns[-1]
            self.addColumn(newColumn,row.routeCost,_name)
            self.init_routes_df[_name] = row.colDF.iloc[:,-1]

    def shortCuttingColumns(self,_var_keywords = 'DP'):
        _cols = self.init_routes_df.set_index('labels')
        DP_cols = _cols.loc[:,_cols.columns.str.contains(_var_keywords)].copy()
        count = 0 
        sc_col_count = 0
        t1 = time.time()
        for r_name, col in DP_cols.items():
            if ((count/len(DP_cols.columns))*100 % 10)==0:
                print('shortcutting columns:',count,'/',len(DP_cols.columns))
            if (col[(col>=2)&(col.index.isin(self.nodes))].size >=1):
#                 print(r_name)
                node_seq = self.DPRouteDict[r_name]
                sct_seq = [self.depot[0]]; visited_seq = [self.depot[0]]
                for n in node_seq:
                    if n not in visited_seq: 
                        sct_seq.append(n)
                        visited_seq.append(n)
                sct_seq.append(self.depot[0])  
                arc_route = [(sct_seq[i],sct_seq[i+1]) for i in range(len(sct_seq)-1)]
#                 print(node_seq,sct_seq)
                qr = sum([self.customer_demand[c] for c in sct_seq])
                lr = self.calculateLr(arc_route)
                _c_tsp = (lr*qr)/(2*col['m'])
                _route_coef = np.array(sct_seq+arc_route,dtype=object)
                sc_col = pd.Series(index = _cols.index,data=0,name=r_name)
                sc_col.loc[_route_coef]+=1
                sc_col.loc['m'] = col['m']
                sc_col.loc['lr'] = lr
#                 print(sc_col)
#                 return sc_col
                _c_trp = self.calculateCostOfRoute(sc_col)['dem_weighted_distance']
                _cost = _c_tsp+_c_trp
#                 print(r_name,'OldCost:',self.model.getVarByName(r_name).Obj ,'SCTCost:',_cost)
                #Update DF: Use sc_col for updating init_routes_df
                self.model.getVarByName(r_name).Obj = _cost
                self.model.update()
                self.init_routes_df.loc[:,r_name] = sc_col.values
                sc_col_count+=1
            count+=1
        self.shortcutCols = sc_col_count
        self.shortcutColsPc = len(DP_cols.columns)
        self.shortcutColsTe = time.time()-t1
            
    def addVehicleToColumns(self,_M,_var_keywords = None,_mode = "New"):
        if _var_keywords is None: _var_keywords = ""
        _cols = self.init_routes_df.set_index('labels')
        pc_count = 0 
        gen_count = 0
        print("Adding vehicle and generating new columns, mode:",_mode)
        if _mode == "New":
            DP_cols = _cols.loc[:,_cols.columns.str.contains(_var_keywords)].copy()
            for r_name, col in DP_cols.items():
                if ((pc_count/len(DP_cols.columns))*100 % 10)==0:
                    print('processing columns:',pc_count,'/',len(DP_cols.columns)) 
                if ("DP" in r_name):
                    node_seq = list(set(self.DPRouteDict[r_name]))
                    qr = sum([self.customer_demand[c] for c in node_seq])
                    lr = col['lr']
                    _c_trp = self.calculateCostOfRoute(col)['dem_weighted_distance']
                    for m_ass in range(int(col['m'])+1,_M+1):
                        new_col_name = r_name+"-ma"+str(m_ass)
                        new_col = col.copy()
                        new_col.loc['m'] = m_ass
                        _c_tsp = (lr*qr)/(2*m_ass)
                        _cost = _c_tsp+_c_trp
    #                     print(col[self.customer_index],col['m'])
                        _temp_col = list(col[self.customer_index].values)+ [m_ass]
                        newColObj = Column(_temp_col, self.model.getConstrs())

                        self.addColumn(newColObj,_cost,new_col_name)
                        self.model.update()
                        self.init_routes_df.loc[:,new_col_name] = new_col.values
                        gen_count+=1
                else: #route from initial set
                    if col['m'] == self.constant_dict['max_vehicles_proute']: 
                        node_seq = pd.Series(col[self.customer_index][col>=1].index)
                        qr = sum([self.customer_demand[c] for c in node_seq])
                        lr = col['lr']
                        old_cost = self.model.getVarByName(r_name).Obj
                        for m_ass in range(int(col['m'])+1,_M+1):
                            new_col_name = r_name+"-ma"+str(m_ass)
                            new_col = col.copy()
                            new_col.loc['m'] = m_ass
                            update_term = -lr*qr*0.5*(m_ass-col['m'])/(col['m']*m_ass)
                            _cost = old_cost+update_term
    #                         print(col[self.customer_index].values,col['m'])
                            _temp_col = list(col[self.customer_index].values)+ [m_ass]
                            newColObj = Column(_temp_col, self.model.getConstrs())

                            self.addColumn(newColObj,_cost,new_col_name)
                            self.model.update()
                            self.init_routes_df.loc[:,new_col_name] = new_col.values
                            gen_count+=1
                pc_count+=1
            print("Finish generating new: {0} columns from original {1} columns".format(gen_count,pc_count))
        elif _mode == "Continue": #Previously added columns with max_m = _M-1
            DP_cols = _cols.loc[:,_cols.columns.str.contains("ma")].copy()
            for r_name, col in DP_cols.items():
                if ((pc_count/len(DP_cols.columns))*100 % 10)==0:
                    print('processing columns:',pc_count,'/',len(DP_cols.columns))
                if col['m'] == _M-1: 
                    node_seq = pd.Series(col[self.customer_index][col>=1].index)
                    qr = sum([self.customer_demand[c] for c in node_seq])
                    lr = col['lr']
                    old_cost = self.model.getVarByName(r_name).Obj
                    for m_ass in range(int(col['m'])+1,_M+1):
                        new_col_name = r_name.replace('ma%s'%str(m_ass-1),'ma%s'%str(m_ass))
                        new_col = col.copy()
                        new_col.loc['m'] = m_ass
                        update_term = -lr*qr*0.5*(m_ass-col['m'])/(col['m']*m_ass)
                        _cost = old_cost+update_term
    #                         print(col[self.customer_index].values,col['m'])
                        _temp_col = list(col[self.customer_index].values)+ [m_ass]
                        newColObj = Column(_temp_col, self.model.getConstrs())

                        self.addColumn(newColObj,_cost,new_col_name)
                        self.model.update()
                        self.init_routes_df.loc[:,new_col_name] = new_col.values
                        gen_count+=1
                pc_count+=1
            print("Finish generating new: {0} columns from original {1} columns".format(gen_count,pc_count))
                        
    def calculateLr(self, route_arcs):
        lr = self.fixed_setup_time+(pd.Series(route_arcs).apply(lambda x:self.distance_matrix[x]).sum()/self.truck_speed)
#         print(route_arcs,lr)
        return lr

    def runColumnsGeneration(self,_m_collections,_pricing_status=False,
                             _check_dominance=True,_dominance_rule=None,
                             _DP_ver=None,_time_limit=None,_filtering_mode=None,
                            _bch_cond=None,_node_count_lab=None,_acc_flag=None):
        outer_dict = dict(zip(['Duals','Inner','ttTime','ttStates'],[[],[],[],[]]))
        inner_dict = dict(zip(['m','route','reward','#states','time'],[None,None,None,None,None]))
        if _DP_ver not in ['ITER_M','SIMUL_M']:
            print("Invalid DP Mode")
        else:
            print('.Running Col. Gen. with DP mode: ', _DP_ver, "Dom Rule:",_check_dominance,_dominance_rule )
            print('| Max nodes visited: %s'%self.max_nodes_proute_DP,'| Max vehicles per route: %s'%self.max_vehicles_proute_DP)
            if (_DP_ver == "ITER_M"):
                print("Dominance Checking:",_check_dominance,', rule:',_dominance_rule)
                self.solveRelaxedModel()
                duals_vect = pd.DataFrame(self.getDuals(), index = self.customers + ['m'])
                opt_cond_vect = pd.Series(index = _m_collections,data=False)
                out_loop_counter = 0
                t1 = time.time()
                iter_log = dict()
                self.colgenLogs = dict()
                iter_log['es_time'] = 0; iter_log['duals'] = duals_vect[0]; iter_log['cols_gen'] = 0;
                iter_log['cols_add'] = 0; iter_log['max_stops'] = 0;
                self.colgenLogs[out_loop_counter]=iter_log
                self.feasibleStatesExplored = 0
                _outerLogList = []
                t1 = time.time()
                while opt_cond_vect.sum()<len(_m_collections):
                    iter_log = dict(); proc_list = [];
                    iter_log['es_time'] = time.time()
                    iter_log['cols_gen'] = 0; iter_log['cols_add'] = 0; iter_log['max_stops'] = 0
                    _outerLog = outer_dict.copy()
                    _innerLogList = []
                    for _m_veh in _m_collections:
                        _innerLog = inner_dict.copy()
                        if _pricing_status:
                            print('.Running Col. Gen. for m_r:', _m_veh,'| Max nodes visited: %s'%self.max_nodes_proute_DP, '| Out-loop-%s'%out_loop_counter)
                        n = len(self.customers)
                        Q = [0]+list(self.customer_demand.loc[self.customers].values)
    #                     M = 3 #max m per route from collection of m
                        print("\n DUALS:",self.getDuals())
                        s0 = self.fixed_setup_time
                        _inner_t = time.time()
                        S,_st_counter = prizeCollectingDP(n,self.cost_matrix,Q,_m_veh,self.getDuals(),s0,
                                                          _veh_cap=self.vehicle_capacity,
                                              _chDom=_check_dominance,_stopLim=self.max_nodes_proute_DP)
                        self.feasibleStatesExplored +=_st_counter[0]
                        _inner_t = time.time()-_inner_t
                        P,bestState = pathReconstruction(S,Q,self.cost_matrix)
    #                     print(S);return P,bestState,S
                        reward = bestState[4]
                        ## Filtering columns
                        if (reward>0.000001) and (bestState[0]>0):
                            dual_r = sum([self.getDuals()[i-1] for i in P[1:-1]])
                            route_cost = -reward+dual_r+_m_veh*self.getDuals()[-1]
                            prx_route = ['O']+['c_%s'%(x) for x in P[1:-1]]+['O']
                            arc_route = [(prx_route[i],prx_route[i+1]) for i in range(len(prx_route)-1)]
                            col_coeff = prx_route+arc_route
                            nCol = pd.DataFrame(self.init_routes_df.set_index('labels').index, columns=['labels'])
                            prefix = str(_m_veh)+str(out_loop_counter)
                            name = 'sDP_C%s-%s'%(prefix,bestState[7])
                            nCol[name] = 0
                            nCol.loc[nCol.labels=='m',name] = _m_veh
                            nCol.loc[nCol.labels=='lr',name] = sum([self.distance_matrix[tup]/self.truck_speed for tup in arc_route])
                            print(bestState,'M:',_m_veh)
                            print('PrxRoute:',prx_route,'ArcRoute:',arc_route)
                            print('Route:',P,'RouteCost:',route_cost,'Reward:',reward)
                            self.DPRouteDict[name] = prx_route
                            # nCol
                            for idx in col_coeff:
                                nCol.loc[nCol.labels==idx,name] +=1
                            adColDf = pd.DataFrame(columns=['routeCost','colDF'])
                            adColDf.loc[name,['routeCost']] =[route_cost]
                            adColDf.loc[name,['colDF']] = [nCol]
                            #Add columns
    #                         return nCol,adColDf
                            self.generateColumns(adColDf, duals_vect)
                            iter_log['cols_add'] +=1
                            _innerLog['m'] = _m_veh; _innerLog['route'] = P;_innerLog['reward'] = reward
                            _innerLog['#states'] = _st_counter;_innerLog['time'] = _inner_t; 
                            _innerLogList=_innerLogList+[_innerLog]
                        else: 
                            opt_cond_vect[_m_veh] = True
                            _innerLog['m'] = _m_veh; _innerLog['#states'] = _st_counter
                            _innerLog['time'] = _inner_t; _innerLogList=_innerLogList+[_innerLog]
                            continue
                        tt_states = sum([len(l) for l in S])
                        iter_log['cols_gen'] += tt_states
                    #Resolve relax model
                    self.solveRelaxedModel()
                    duals_vect = pd.DataFrame(self.getDuals(), index = self.customers + ['m'])
                    out_loop_counter+=1
                    iter_log['es_time'] = time.time()-iter_log['es_time']
                    iter_log['duals'] = duals_vect[0]
                    self.colgenLogs[out_loop_counter]=iter_log
                    ######COMPARISON##########
                    _outerLog['ttTime'] = sum([nn['time'] for nn in _innerLogList])
                    _outerLog['ttStates'] = np.sum([nn['#states'] for nn in _innerLogList],axis=0)
                    _outerLog['Duals'] = self.getDuals()
                    _outerLog['Inner'] = _innerLogList
                    _outerLogList = _outerLogList+[_outerLog]
                self.colGenTe = time.time()-t1
                self.colGenCompLog = _outerLogList
                print('Col.Gen. Completed!...Elapsed-time:',self.colGenTe)
            elif (_DP_ver == "SIMUL_M"):
                self.solveRelaxedModel()
                duals_vect = pd.DataFrame(self.getDuals(), index = self.customers + ['m'])
                opt_cond = False; out_loop_counter = 0; iter_log = dict();self.colgenLogs = dict()
                iter_log['es_time'] = 0; iter_log['duals'] = duals_vect[0]; iter_log['cols_gen'] = 0;
                iter_log['cols_add'] = 0; iter_log['max_stops'] = 0;
                self.colgenLogs[out_loop_counter]=iter_log; self.feasibleStatesExplored = 0
                t1 = time.time(); _outerLogList = []
                if _acc_flag is not None: 
                    in_duals = np.array([0]*len(self.customers + ['m']))
                    out_duals = np.array(self.getDuals())
                    primal_bound = self.relaxedModel.ObjVal
                    dual_bound = 0
                    print("\n ==== Using In-out acc. method. Set coeff to:", _acc_flag)
                    print("In dual:", in_duals)
                    print("Out dual:", out_duals)
                    print("---------------------------------")
                    print("Primal bound:", primal_bound)
                    print("Dual bound:", dual_bound)
                    print("---------------------------------")
                    
#                     out_duals = np.array(self.getDuals())
#                     conv_comb_duals = in_duals*(_acc_flag)+out_duals*(1-_acc_flag)
                    
                while not(opt_cond):
                    iter_log = dict(); proc_list = [];
                    _outerLog = outer_dict.copy(); _innerLog = inner_dict.copy()
                    iter_log['es_time'] = time.time(); iter_log['cols_gen'] = 0; 
                    iter_log['cols_add'] = 0; iter_log['max_stops'] = 0
                    
                    n = len(self.customers)
                    Q = [0]+list(self.customer_demand.loc[self.customers].values)
                    s0 = self.fixed_setup_time
                    if _acc_flag is not None: 
                        print("Primal bound:", primal_bound); print("Dual bound:", dual_bound)
                        if (primal_bound-dual_bound)<epsilon: 
                            print("\n ==== GAP LESS THAN EPSILON, OPTIMAL")
#                             print("In dual:", in_duals)
#                             print("Out dual:", out_duals)
#                             print("---------------------------------")
                            print("Primal bound:", primal_bound)
                            print("Dual bound:", dual_bound)
                            print('Feasible States explored in {}-iters:{}'.format(out_loop_counter,
                                                                                   self.feasibleStatesExplored))
                            print("---------------------------------")
                            opt_cond = True;  continue;
                        elif (dual_bound>1e6):
                            print("\n ==== INFEASIBLE NODE")
                            opt_cond = True; continue;
                        # Duals being used
                        _duals = in_duals*(_acc_flag)+out_duals*(1-_acc_flag)
                        print("\n ==== GAP STILL LARGER THAN EPSILON")
                        print("In dual:", in_duals)
                        print("Out dual:", out_duals)
                        print("CONVEX COMB DUAL:", _duals)
#                         print("---------------------------------")
                    else:
                        if _pricing_status:
                            print(' Out-loop-%s'%out_loop_counter)
                            print(' Start Running DP','| Max nodes visited: %s'%self.max_nodes_proute_DP,'| Max vehicles per route: %s'%self.max_vehicles_proute_DP,)
                            print(' Solving time limit set to:',_time_limit,'secs.',"Dominance Checking:",_chDom,"Domination Rule:", _dom_rule)
                            
                        # Duals being used
                        _duals = self.getDuals()
#                     print("\n DUALS:",_duals)
                    _inner_t = time.time()
                    S,_st_counter = prizeCollectingDPVer2(
                                n,self.cost_matrix,Q,
                                _duals,s0,_mprDP=self.max_vehicles_proute_DP,
                                _ttM=self.max_vehicles,
                                _veh_cap=self.vehicle_capacity,
                                _chDom=_check_dominance,
                                _stopLim=self.max_nodes_proute_DP,
                                _domVer=_dominance_rule,
                                _time_limit=_time_limit)

                    self.feasibleStatesExplored +=_st_counter[0];
                    _inner_t = time.time()-_inner_t
                    PList,bestStateList = pathReconstructionCTCVer2(S,Q,
                                            self.cost_matrix,_filtering_mode,self.max_vehicles_proute_DP,
                                            _bch_cond=_bch_cond)
                    rwdList = [((b[5]>0.000001) and (b[0]>0)) for b in bestStateList]
                    _innerLogList=[]; 
#                     print(bestStateList)
                    if _acc_flag is not None: 
                        if not(any(rwdList)): # Dual is feasible, so no improved route can be found!
                            # Update in-dual and dual-bound
                            in_duals = _duals
                            dual_bound = sum(_duals[:-1])+self.max_vehicles*_duals[-1]
#                             print("\n ==== NOT FOUND IMPROVEMENT, DUAL FEASIBLE, UPDATE IN-DUAL")
#                             print("In dual:", in_duals)
#                             print("Out dual:", out_duals)
#                             print("---------------------------------")
#                             print("Primal bound:", primal_bound)
#                             print("Dual bound:", dual_bound)
#                             print("---------------------------------")
                            continue # start the new DP with new dual
                        else: # Dual is infeasible, so update out-dual
                            pass
#                             out_duals = _duals
                            # Then, add new columns and update the primal bound
                            
                    else:
                        if not(any(rwdList)): opt_cond = True; continue;
                    for idx in range(len(bestStateList)):
                        _innerLog = inner_dict.copy()
                        P = PList[idx];bestState = bestStateList[idx]
                        reward = bestState[5]
                        ## Filtering columns
                        if (reward>0.000001) and (bestState[0]>0):
                            dual_r = sum([_duals[i-1] for i in P[1:-1]]) #count repeat visits
                            route_cost = -reward+dual_r+bestState[4]*_duals[-1]
                            prx_route = ['O']+['c_%s'%(x) for x in P[1:-1]]+['O']
                            arc_route = [(prx_route[i],prx_route[i+1]) for i in range(len(prx_route)-1)]
                            col_coeff = prx_route+arc_route
                            nCol = pd.DataFrame(self.init_routes_df.set_index('labels').index,
                                                columns=['labels'])
                            if _node_count_lab is not None: prefix ="BnP%s-"%(_node_count_lab)+ str(idx)+str(out_loop_counter)
                            else: prefix = str(idx)+str(out_loop_counter)
                            name = 'sDP_C%s-%s'%(prefix,bestState[7])
                            nCol[name] = 0
                            nCol.loc[nCol.labels=='m',name] = bestState[4]
                            nCol.loc[nCol.labels=='lr',name] = sum([self.distance_matrix[tup]/self.truck_speed for tup in arc_route])
                            print(bestState,'RouteCost:',route_cost,'RouteName:',name)
                            print('Route:',P,'M:',bestState[4],'Reward:',reward)
                            self.DPRouteDict[name] = prx_route
                            # nCol
                            for idx in col_coeff:
                                nCol.loc[nCol.labels==idx,name] +=1
                            adColDf = pd.DataFrame(columns=['routeCost','colDF'])
                            adColDf.loc[name,['routeCost']] =[route_cost]
                            adColDf.loc[name,['colDF']] = [nCol]
                            #Add columns
                            self.generateColumns(adColDf, _duals)
                            iter_log['cols_add'] +=1
                            _innerLog['m'] = bestState[4]
                            _innerLog['route'] = P
                            _innerLog['reward'] = reward
                            _innerLog['#states'] = None
                            _innerLog['time'] = None
                            _innerLogList=_innerLogList+[_innerLog]
                        else:
                            _innerLog['m'] = bestState[4]
                            _innerLog['route'] = None
                            _innerLog['reward'] = None
                            _innerLog['#states'] = None
                            _innerLog['time'] = None
                            _innerLogList=_innerLogList+[_innerLog]
                    tt_states = sum([len(l) for l in S])
                    iter_log['cols_gen'] += tt_states 
                    #Resolve relax model
                    self.solveRelaxedModel()
                    if _acc_flag is not None: 
                        primal_bound = self.relaxedModel.ObjVal
                        out_duals = np.array(self.getDuals())
#                         print("\n ==== NOT FOUND IMPROVEMENT, DUAL FEASIBLE, UPDATE IN-DUAL")
#                         print("In dual:", in_duals)
#                         print("Out dual:", out_duals)
#                         print("---------------------------------")
#                         print("Primal bound:", primal_bound)
#                         print("Dual bound:", dual_bound)
#                         print("---------------------------------")
#                     duals_vect =  pd.DataFrame(self.getDuals(), index = self.customers + ['m'])
                    ######COMPARISON##########
                    _outerLog['ttTime'] = _inner_t
                    _outerLog['ttStates'] = _st_counter
                    _outerLog['Duals'] = self.getDuals()
                    _outerLog['Inner'] = _innerLogList
                    _outerLogList = _outerLogList+[_outerLog]
                    out_loop_counter+=1
                    iter_log['es_time'] = time.time()-iter_log['es_time']
                    iter_log['duals'] = _duals[0]
                    self.colgenLogs[out_loop_counter]=iter_log
                self.colGenTe = time.time()-t1
                self.colGenCompLog = _outerLogList
                print('Col.Gen. Completed!...Elapsed-time:',self.colGenTe)
 
    def calculateAverageRemainingSpace(self,_model_vars,):
        vars_value = pd.Series(_model_vars)
        sol_vec = pd.DataFrame(index = vars_value.apply(lambda x:x.VarName))
        sol_vec['value'] = vars_value.apply(lambda x:x.X).values
        optimal_routes_name_list = sol_vec.loc[sol_vec['value']>=0.98].index.to_list()
        _cumulative_space = 0
        for j in range(len(optimal_routes_name_list)):
            _route_name = optimal_routes_name_list[j]
            _cumulative_space+=self.getRemainingSpace(_route_name)
        _avg_rem_space = _cumulative_space/self.max_vehicles
        return _avg_rem_space
        
    def getRemainingSpace(self, _route_name):
        '''Absolute remaining space for all mr'''
        ref_df = self.init_routes_df.set_index('labels')
        _col = ref_df.loc[:][_route_name]
        _mr = _col['m']
        node_seq = pd.Series(_col[self.customer_index][_col>=0.7].index)
        _qr = sum([self.customer_demand[c] for c in node_seq])
        _lr = _col['lr']
        _abs_rem_space = (_mr*self.vehicle_capacity) - (_lr*_qr)
        print(_route_name,', Rem. space=',_abs_rem_space,', mr=',_mr)
        return _abs_rem_space
        
    def getRoute4Plot(self, _route_name_list, _colums_df,_route_config):
        reformatted_arcs=[]
        ref_df = self.init_routes_df.set_index('labels')
        COLORLIST = ["#FE2712","#347C98","#FC600A",
                     "#66B032","#0247FE","#B2D732",
                    "#FB9902","#4424D6","#8601AF",
                    "#FCCC1A","#C21460","#FEFE33"]
        content_array = ['arcs_list','config','route_info','info_topics',
                         'column_width','column_format']
        route_info_topic_array = ['lr','total_demand','demand_waiting','avg_waiting_per_pkg','pkgs','utilization']
        column_width = [3,3,3.5,3,3.2,3.2]
        column_format = ['.2f','.2f',None,'.2f','.2f','.2f']
        for j in range(len(_route_name_list)):
            idx = _route_name_list[j]
            col_idx = j%12
#             print(idx)
            curr_route_config = _route_config.copy()
            curr_route_config['line_color'] = COLORLIST[col_idx]
            sample_r = _colums_df.loc[:][idx]
            curr_route_config['name'] = idx+"-"+str(round(_colums_df.loc['m'][idx]))+"m"
            sample_arcs = sample_r[sample_r.index.isin(self.arcs)][sample_r==1]
            sample_nodes = sample_r[sample_r.index.isin(self.customers)][sample_r>=1]
            #Route INFO:
            _qr = sum([self.customer_demand[c] for c in sample_nodes.index.to_list()])
            _mr = round(_colums_df.loc['m'][idx])
            cost_dict = self.calculateCostOfRoute(sample_r)
            _avg_CTC_cost = cost_dict['total_cost']
            ############
            _lr = sample_r.loc['lr']
            _dem_waiting = cost_dict['dem_waiting']
            _avg_waiting_per_pkg = _avg_CTC_cost/_qr
            _pkgs = _lr*(_qr)
            _util = (self.vehicle_capacity*_mr-self.getRemainingSpace(idx))*100/(self.vehicle_capacity*_mr)
            ###########
            route_info_value = [_lr,_qr,_dem_waiting,_avg_waiting_per_pkg,_pkgs,_util]
            route_info_dict = dict(zip(route_info_topic_array,route_info_value))
            route_plot_dict = dict(zip(content_array,
                           [sample_arcs.index.to_list(),curr_route_config,
                            route_info_dict,route_info_topic_array,column_width,column_format]))
            reformatted_arcs += [route_plot_dict]
        return reformatted_arcs    
    
    def getRouteSolution(self,_model_vars,_edge_plot_config,_node_trace,_cus_dem):
        vars_value = pd.Series(_model_vars)
        sol_vec = pd.DataFrame(index = vars_value.apply(lambda x:x.VarName))
        sol_vec['value'] = vars_value.apply(lambda x:x.X).values
        optimal_routes = sol_vec.loc[sol_vec['value']>=0.98]
        ref_df = self.init_routes_df.set_index('labels')
#         print(ref_df.loc[['m','lr']][optimal_routes.index])
        formatted_routes_list =  self.getRoute4Plot(optimal_routes.index.to_list(),
                                                                ref_df,_edge_plot_config)
        return formatted_routes_list
     
    def plotCurrentSolution(self,_model_vars,_edge_plot_config,_node_trace,_title,_cus_dem):
        vars_value = pd.Series(_model_vars)
        sol_vec = pd.DataFrame(index = vars_value.apply(lambda x:x.VarName))
        sol_vec['value'] = vars_value.apply(lambda x:x.X).values
        optimal_routes = sol_vec.loc[sol_vec['value']>=0.98]
        ref_df = self.init_routes_df.set_index('labels')
        print(ref_df.loc[['m','lr']][optimal_routes.index])
        formatted_routes_list =  self.getRoute4Plot(optimal_routes.index.to_list(),
                                                                ref_df,_edge_plot_config)
        vis_sol.plot_network(formatted_routes_list,_node_trace,_title,_cus_dem)
    
        
        
def prizeCollectingDPVer2(_n,_C,_Q,_dual,_s0,_veh_cap,_mprDP,_ttM,_chDom=True,_stopLim=None,_domVer=None,
                          _time_limit=None):
    if _stopLim is None: _stopLim = _n
    if _time_limit is None: _time_limit = np.inf; 
    print('Solving time limit set to:',_time_limit,'secs.',"Dominance Checking:",_chDom)
    if _mprDP is None: _mprDP = _ttM
    print('Setting max-vehicles-proute DP to:',_mprDP)
    print('Setting max-stops-proute DP to:',_stopLim)
    if _chDom:
        if (_domVer is None): print("Please specify dominance version! _domVer is ",_domVer)
        elif (_domVer not in [2]): print("Wrong input of dominance version! _domVer should be in",[1,2,3])

    _counter = 0; _rch_counter = 0; _h0 = 0
    # Def I: L = [i,d,l,p,m,rwd,pFlag,prevN,counter]
    # L0 = [0,0,0,0,0,0,False,None,_counter] 
    # Linf = [np.inf,np.inf,np.inf,np.inf,-np.inf,True,None,np.inf]

    L0 = [0,0,0,0,0,0,False,None,_counter,_h0] 
    Linf = [np.inf,np.inf,np.inf,np.inf,-np.inf,True,None,np.inf,np.inf]
    S = [[L0]]+[[[x+1]+Linf] for x in range(_n)]; _mu = _dual[-1]
    tFlag = False
    _time = time.time()
    _neg_cost_counter=0
    while not(tFlag):
        for i in range(_n+1):
            #TIME limit checking
            if (time.time()-_time)>_time_limit:
                print('Reach time limit!! ','{0}/{1}'.format(time.time()-_time,_time_limit))
                print('Rwd each state:',[S[i][0][5] for i in range(_n+1)])
                pos_rwd_exist = any([s[5]>0.000001 for s in [S[i][0] for i in range(_n+1)]])
                if pos_rwd_exist: tFlag=True;break;
                else: 
                    _neg_cost_counter+=1;
                    if _neg_cost_counter>=5: tFlag=True;break;
                    print(_neg_cost_counter,'No positive reduced cost found! Reset time & continue searching... ','states explored:',sum([len(x) for x in S]));_time = time.time()
            for w in range(len(S[i])):
                if not(S[i][w][6]):
                    if S[i][w][3] >= _stopLim:
                        S[i][w][6] = True
                    else:
                        # Avoid forbidden link (0-branch)
                        reachTo = [x for x in range(1,_n+1) if ((x!=i) and (x!=S[i][w][7]))] 
                        S[i][w][6] = True
                        for j in reachTo:
                            _rch_counter+=1
                            _d = S[i][w][1] + _Q[j]
                            _l = S[i][w][2] + _C[(i,j)]
                            _p = S[i][w][3] + 1
                            
                            if (_mprDP*_veh_cap < (_d*(_l+_C[(j,0)]))):
                                # Biggest m is infeasible
                                continue
                            else:
                                _phi = np.sqrt((_d)*(_l+_C[(j,0)])/(2*np.abs(_mu)))
                                _m_opt = getOptM(_phi,_mprDP, _d, _l+_C[(j,0)], _mu)    
                                # Optimal m is greater than DP limit, set it equal to the limit
                                if _m_opt>_mprDP: _m_opt = _mprDP
                                # Optimal m is less than or equal to DP limit
                                else:
                                    _serving_cap_feas = (_m_opt*_veh_cap >= (_d*(_l+_C[(j,0)])))
                                    # If Opt m is not demand feasible, increase until feasible or reaching the DP limit
                                    while (not(_serving_cap_feas) and (_m_opt < _mprDP)):
                                        _m_opt+=1
                                        _serving_cap_feas = (_m_opt*_veh_cap >= (_d*(_l+_C[(j,0)])))

                            # Only allows demand feasible state to be constructed
                            if (_m_opt*_veh_cap >= (_d*(_l+_C[(j,0)]))):
                                _h_opt = h_func(_m_opt, _d, _l+_C[(j,0)], _mu);
                                _rwd = S[i][w][5] + transRwdVer2(S[i][w],j,_C,_Q,_m_opt,_dual,_s0)
                                nS = [j,_d,_l,_p,_m_opt,_rwd,False,i]
                                _counter+=1
                                nS = nS+[_counter,_h_opt]
                                if _chDom: 
                                    if _domVer==2: S = checkDominanceCTCVer2(nS,S,_C,_mu)
                                    elif _domVer==3: S = checkDominanceCTCVer3(nS,S,_C,_mu,_mprDP)
                                    else: print("ERROR: WRONG DOM. VER."); return;
                                else: S[j].append(nS)
                            S[i][w][6] = True
            _temp = sorted(S[i], key=itemgetter(5),reverse=True)
            S[i] = _temp
        _all_pc = True #All processed
        _b_pt = False #Break point
        for i in range(_n+1):
            for l in range(len(S[i])):
                if not(S[i][l][6]): _b_pt=True;break;
            if _b_pt:_all_pc=False;break
        if _all_pc: tFlag=True
    return S,(_counter,_rch_counter)

def h_func(_phi, _q, _lr, _mu):
    if _phi>0: return (_mu*(_phi)) - ((_q*_lr)/(2*_phi))
    else: return 0

def getOptM(_phi,_M, _q, _lr, _mu):
    cphi = np.ceil(_phi); fphi = np.floor(_phi);
    cel_h = h_func(cphi, _q, _lr, _mu);flr_h = h_func(fphi, _q, _lr, _mu);
    if (cel_h >= flr_h) and (cphi <= _M):
        opt_m = cphi; 
    elif (cel_h < flr_h) and (fphi <= _M):
        opt_m = fphi; 
    else:
        opt_m = _M; 
    return opt_m

def transRwdVer2(_cS,_j,_C,_Q,_opt_m,_dual,_s0):
    _i = _cS[0]
    _nD = _cS[1]+_Q[_j]
    _nL = _cS[2]+ _C[(_i,_j)]
    _nPi = _dual[_j-1] #dual's index doesnt have depot
    _nM = _opt_m
    _mu = _dual[-1]
    _reCostI = _C[(_i,0)]
    _reCostJ = _C[(_j,0)]
    if (_cS[4]==0):
        _tRwd = _nPi +_mu*(_nM)+\
                - (_Q[_j]*(_cS[2]+_C[(_i,_j)])) \
                - ((0.5/_nM)*_nD*(_cS[2]+_C[(_i,_j)]+_reCostJ+_s0))
    else:
        _tRwd = _nPi +_mu*(_nM-_cS[4])+\
                - (_Q[_j]*(_cS[2]+_C[(_i,_j)])) \
                - ((0.5/_nM)*_nD*(_cS[2]+_C[(_i,_j)]+_reCostJ+_s0))\
                + ((0.5/_cS[4])*_cS[1]*(_cS[2]+_reCostI+_s0))
    return _tRwd

def checkDominanceCTCVer2(_nS,_S,_C,_mu):
    i = _nS[0]; adFlag = True;
    h_nS = _nS[9]
    for w in range(len(_S[i])-1,-1,-1):
        cpS = _S[i][w]
        h_cpS = cpS[9]
        if ((_nS[1]<=cpS[1])and(_nS[2]<=cpS[2])and(_nS[3]<=cpS[3])and(_nS[5]+(h_cpS-h_nS)>=cpS[5])):
            del _S[i][w]
        elif ((_nS[1]>=cpS[1])and(_nS[2]>=cpS[2])and(_nS[3]>=cpS[3])and(_nS[5]+(h_cpS-h_nS)<=cpS[5])):
            adFlag = False #Throw away _nS
            break
    if adFlag:
        _S[i].append(_nS)
    return _S

def checkDominanceCTCVer3(_nS,_S,_C,_mu,_mprDP):
    #dominance with m_cap
    i = _nS[0]; adFlag = True;
    h_nS = _nS[9]; m_nS = _nS[4]
    for w in range(len(_S[i])-1,-1,-1):
        cpS = _S[i][w]; h_cpS = cpS[9]; m_cpS = cpS[4]
        if ((m_nS==m_cpS) and (m_nS==_mprDP)): #Only Za>Zb is sufficent
        # if ((m_nS==m_cpS)) : #For test! & it's wrong!
            if ((_nS[1]<=cpS[1])and(_nS[2]<=cpS[2])and(_nS[3]<=cpS[3])and(_nS[5]>=cpS[5])):
                del _S[i][w]
            elif ((_nS[1]>=cpS[1])and(_nS[2]>=cpS[2])and(_nS[3]>=cpS[3])and(_nS[5]<=cpS[5])):
                adFlag = False #Throw away _nS
                break
        else: #Need correction terms 
            if ((_nS[1]<=cpS[1])and(_nS[2]<=cpS[2])and(_nS[3]<=cpS[3])and(_nS[5]+(h_cpS-h_nS)>=cpS[5])):
                del _S[i][w]
            elif ((_nS[1]>=cpS[1])and(_nS[2]>=cpS[2])and(_nS[3]>=cpS[3])and(_nS[5]+(h_cpS-h_nS)<=cpS[5])):
                adFlag = False #Throw away _nS
                break
    if adFlag:
        _S[i].append(_nS)
    return _S


def pathReconstructionCTCVer2(_S,_Q,_C,_filtering_mode=None,_mprDP=None,_bch_cond=None):
    if _filtering_mode is None: _filtering_mode = "BestRwdPerI"
    if _filtering_mode not in ["BestRwdPerI","BestRwdPerM"]: print("Incorrect filtering mode!")
#     print(' Filtering Mode:',_filtering_mode)
    _route_list=[];_bestSt_list=[]
    _dummy_s = [0,0,0,0,0,-np.inf,True,None,None]
    
     ####### Data from braching cond #####
    _abandon_n_dict = dict();
    forbid_link=[]; necess_link=[]; skip_ending_n=[];
    if _bch_cond is not None: 
        for bh in _bch_cond:
            if (bh[1]==0): 
                forbid_link+=[bh[0]] # 0-branch
                if bh[0][1]=="O":
                    skip_ending_n+=[int(bh[0][0].split("_")[-1])]
            if (bh[1]==1): 
                necess_link+=[bh[0]] # 1-branch
        forbid_link = [bh[0] for bh in _bch_cond if (bh[1]==0)] 
        necess_link = [bh[0] for bh in _bch_cond if (bh[1]==1)] 
#     print("skip_ending_n",skip_ending_n)
#     print("bch-conds:",_bch_cond)
#     print("forbidden_link:",forbid_link)
#     print("necessary_link:",necess_link)
    ####################################
    
    
    if _filtering_mode == "BestRwdPerM":
        _temp_S = [_S[i][:-1] for i in range(len(_S))]
        _maxMPerI=[]
        for i in range(len(_S)):
            if len(_temp_S[i])==0:
                _maxMPerI.append(0)
            else:
                _maxMPerI.append(np.max(np.array(_temp_S[i])[:,4],axis=0))
        if _mprDP is None: _mprDP = int(np.max(_maxMPerI))
        _maxMdict = dict(zip([i for i in range(1,_mprDP+1)],[[]]*_mprDP))
        _bestStList = [_S[i][0] for i in range(len(_S))]
        _route_list=[];_bestSt_list=[]
        for _idx in range(1,len(_S)):
            for _m in range(1,int(_mprDP+1)):
                f_list = list(filter(lambda x:(x[4]==(_m)), _S[_idx]))
#                 print(_m,f_list,_maxMdict[_m])
                if (len(f_list)==0): 
                    if len(_maxMdict[_m])==0: 
                        _maxMdict[_m] = [-1,0,0,0,_m,-np.inf,True,None,None]
                    continue
                else:
                    _bestR = sorted(f_list, key=itemgetter(5),reverse=True)[0]
                    if len(_maxMdict[_m])==0:
                        _maxMdict[_m] = _bestR
                    elif _maxMdict[_m][5] <_bestR[5] :
                        _maxMdict[_m] = _bestR
        print(_maxMdict)    
        for _idx in range(1,_mprDP+1):
            _route = [0]
            throw_away_flag = False
            lSt = _maxMdict[_idx]
            if ((len(lSt)==0)): continue
            if ((lSt[0]==0)): continue
            lI = lSt[0]; lD = lSt[1]; lL = lSt[2]
            lP = lSt[3]; lM = lSt[4]; prevN = lSt[7]
            _route = [lI]+_route
            _bestSt = lSt
            while lP!=0:
                f_list = list(filter(lambda x:(x[3]==(lP-1)) and (x[1]==lD-_Q[lI]) and (x[4]<=lM) and ((np.abs(x[2]-(lL-_C[(prevN,lI)]))<0.00000001)), _S[prevN]))
                if len(f_list)==0:
                    print("Unmatched State:",lSt)
                    throw_away_flag=True;break
                else:
                    if len(f_list)>1:
                        print("ALERT!:",f_list)
                        _temp = sorted(f_list, key=itemgetter(2),reverse=False)
                        f_list = _temp[:1]
                    lSt = f_list[0]; lI = lSt[0]; lD = lSt[1]
                    lL = lSt[2]; lP = lSt[3]; lM = lSt[4]
                    prevN = lSt[7]; _route = [lI]+_route
#             print(_idx,_route,_bestSt)
            if (throw_away_flag):
                _route =[] ;_bestSt = [-1,0,0,0,_idx,-np.inf,True,None,None]
            _route_list.append(_route)
            _bestSt_list.append(_bestSt)
                
    elif _filtering_mode == "BestRwdPerI":
        _loop_cond = [True]*len(_S)
        _cter_idx = 0 # node idx
        _order_idx = 0 # First rank of highest reward
        
        _bestStList = [_S[i][0] for i in range(len(_S))]
        while (any(_loop_cond)) and (_cter_idx<len(_S)):
#             print(_loop_cond,_cter_idx,_order_idx)
            _route = [0]
            _route_arcs = []
            throw_away_flag = False
            lSt = _S[_cter_idx][_order_idx]
            if ((lSt[0]==0) or (lSt[7] is None)): 
#                 print("No improvement for state last visit at:",lSt)
                _loop_cond[_cter_idx] = False
                _cter_idx+=1; _order_idx=0
                continue
#                 if _order_idx < len(_S[_cter_idx])-1: 
#                     _order_idx+=1
#                 else:
           
            if (lSt[0] in skip_ending_n):
                print("Skip all state ending at:",lSt[0])
                _loop_cond[_cter_idx] = False
                _cter_idx+=1; _order_idx=0;
                continue
                
            lI = lSt[0]; lD = lSt[1]; lL = lSt[2]
            lP = lSt[3]; lM = lSt[4]; prevN = lSt[7]
            _route = [lI]+_route
            _route_arcs = [(0,lI),(lI,0)]
            _bestSt = lSt
            
            while lP!=0:
                f_list = list(filter(lambda x:(x[3]==(lP-1)) and (np.abs(x[1]-(lD-_Q[lI]))<0.00001) and (x[4]<=lM) and ((np.abs(x[2]-(lL-_C[(prevN,lI)]))<0.00001)), _S[prevN]))
    #             print(f_list)
                if len(f_list)==0:
#                     _f_list = list(filter(lambda x:(x[3]==(lP-1)) and (x[4]<=lM), _S[prevN]))
#                     print("Stop at state:",lSt)
#                     print("Filtered:",_f_list)
                    throw_away_flag=True;break
                else:
                    if len(f_list)>1:
#                         print("ALERT!:",f_list)
                        _temp = sorted(f_list, key=itemgetter(5),reverse=True)
                        f_list = _temp[:1]
                    lSt = f_list[0]
                    lI = lSt[0]; lD = lSt[1]; lL = lSt[2]
                    lP = lSt[3]; lM = lSt[4]; prevN = lSt[7]
                    _route = [lI]+_route;
                    _route_arcs = [(0,lI),(lI,_route_arcs[0][1])]+_route_arcs[1:]
#                 print("Route:",_route,throw_away_flag)

            # check branching conditions
            if not(throw_away_flag):
                for tup in necess_link:
                    tup = (int(tup[0].split("_")[-1].replace("O","0")),int(tup[1].split("_")[-1].replace("O","0")))
                    if tup[0]==0: # there is (i,j) in route where i!=0
                        if (tup not in _route_arcs) and (tup[1] in _route): 
                            throw_away_flag = True; break
                    elif tup[1]==0: # there is (i,j) in route where j!=0
                        if (tup not in _route_arcs) and (tup[0] in _route): 
                            throw_away_flag = True; break
                    else:
                        if (tup not in _route_arcs):
                            throw_away_flag = True; break
    #               
                    if throw_away_flag:  print("Skipped:",_route,"No arc:",tup,"State:",)
            
            # filtering out 0-branch as DP cannot forbid ending at i
            if not(throw_away_flag):
                for tup in forbid_link:
                    tup = (int(tup[0].split("_")[-1].replace("O","0")),int(tup[1].split("_")[-1].replace("O","0")))
                    if (tup in _route_arcs): 
                        print("Skipped:",_route,"Forbidden arc:",tup,"State:",) #_S[_cter_idx][_order_idx])
                        throw_away_flag = True
                        break
    #         print("Route:",_route,throw_away_flag)
            if not(throw_away_flag):
                _route_list.append(_route)
                _bestSt_list.append(_bestSt)
                _loop_cond[_cter_idx] = False
                _cter_idx+=1
                _order_idx=0
            else:
                if _order_idx < len(_S[_cter_idx])-1:
                    _order_idx+=1
                else:
                    _loop_cond[_cter_idx] = False
                    _cter_idx+=1
                    _order_idx=0
    #             print("RouteList:",_route,_bestSt)
#         print(_route_list,_bestSt_list)
    return _route_list,_bestSt_list
    
        
def prizeCollectingDP(_n,_C,_Q,_M,_dual,_s0,_veh_cap,_chDom=True,_stopLim=None):
    if _stopLim is None: _stopLim = _n
    _counter = 0;_rch_counter=0
    L0 = [0,0,0,0,_M*_dual[-1],False,None,_counter]
    S = [[L0]]+[[] for x in range(_n)]
    tFlag = False
    while not(tFlag):
        for i in range(_n+1):
            for l in range(len(S[i])):
                if not(S[i][l][5]):
                    if S[i][l][3] >= _stopLim:
                        S[i][l][5] = True
                    else:
                        reachTo = [x for x in range(1,_n+1) if x!=i] #dont need 0 and i
                        for j in reachTo:
                            _rch_counter+=1
                            _d = S[i][l][1] + _Q[j]
                            _l = S[i][l][2] + _C[(i,j)]
                            _p = S[i][l][3] + 1
                            _rwd = S[i][l][4] + transRwd(S[i][l],j,_C,_Q,_M,_dual,_s0)
                            nS = [j,_d,_l,_p,_rwd,False,i]
                            if checkFeasibility(nS,_veh_cap,_M,_C):
                                _counter+=1
                                nS = nS+[_counter]
                                if _chDom: S = checkDominance(nS,S)
                                else: 
    #                                 print("HEYYY")
                                    S[j].append(nS)
                                    temp = sorted(S[j], key=itemgetter(4),reverse=True)
                                    S[j] = temp
                            S[i][l][5] = True                        
#             print(S)
        _all_pc = True #All processed
        _b_pt = False #Break point
        for i in range(_n+1):
            for l in range(len(S[i])):
                if not(S[i][l][5]): _b_pt=True;break;
            if _b_pt:_all_pc=False;break

        if _all_pc: tFlag=True
    return S,(_counter,_rch_counter)

def checkFeasibility(_nS,_veh_cap, _m, _C):
    _rt_cost = _C[(_nS[0],0)]
    deliver_cap = _nS[1]*(_nS[2]+_rt_cost)
    limit_cap = _veh_cap*_m
#     print('.....FEASIBILITY:','resNode:',_nS[0],'| deliver_cap:',round(deliver_cap,2),'| cap_limit:',round(limit_cap,2))
    if deliver_cap <= limit_cap: return True
    else: return False
    
def transRwd(_cS,_j,_C,_Q,_M,_dual,_s0):
    _i = _cS[0]
    _nD = _cS[1]+_Q[_j]
    _nL = _cS[2]+ _C[(_i,_j)]
    _nPi = _dual[_j-1] #dual's index doesnt have depot
    _reCostI = _C[(_i,0)]
    _reCostJ = _C[(_j,0)]
    _tRwd = _nPi - (_Q[_j]*(_cS[2]+_C[(_i,_j)])) \
                - ((0.5/_M)*_nD*(_cS[2]+_C[(_i,_j)]+_reCostJ+_s0))\
                + ((0.5/_M)*_cS[1]*(_cS[2]+_reCostI+_s0))
    return _tRwd
    
def checkDominance(_nS,_S):
    i = _nS[0]
    adFlag = True
#     print(_S)
    for l in range(len(_S[i])-1,-1,-1):
#         print(l)
        cpS = _S[i][l]
        if ((_nS[1]<=cpS[1])and(_nS[2]<=cpS[2])and(_nS[3]<=cpS[3])and(_nS[4]>cpS[4])):
            del _S[i][l]
        elif ((_nS[1]>=cpS[1])and(_nS[2]>=cpS[2])and(_nS[3]>=cpS[3])and(_nS[4]<cpS[4])):
            adFlag = False #Throw away _nS
            break
    if adFlag:
        _S[i].append(_nS)
    #sort _S[i] by reward
    _temp = sorted(_S[i], key=itemgetter(4),reverse=True)
    _S[i] = _temp
#     print('AfterDominance:',_S[i])
    return _S

def pathReconstruction(_S,_Q,_C):
    _route = [0]
    _temp = [_S[i][0] for i in range(len(_S))]
    lSt = sorted(_temp, key=itemgetter(4),reverse=True)[0]
#     print(sorted(_temp, key=itemgetter(4),reverse=True),lSt)
    if ((lSt[0]==0)): lSt = sorted(_temp, key=itemgetter(4),reverse=True)[1]
    lI = lSt[0]
    lD = lSt[1]
    lL = lSt[2]
    lP = lSt[3]
    prevN = lSt[6]
    _route = [lI]+_route
    _bestSt = lSt
    while lP!=0:
        f_list = list(filter(lambda x:(x[3]==(lP-1)) and (x[1]==lD-_Q[lI]) and ((np.abs(x[2]-(lL-_C[(prevN,lI)]))<0.000001)), _S[prevN]))
#         print(f_list)
        if len(f_list)>1:
            print("ALERT!:",f_list)
            _temp = sorted(f_list, key=itemgetter(4),reverse=True)
            f_list = _temp[:1]
        lSt = f_list[0]
        lI = lSt[0]
        lD = lSt[1]
        lL = lSt[2]
        lP = lSt[3]
        prevN = lSt[6]
        _route = [lI]+_route
    return _route,_bestSt
