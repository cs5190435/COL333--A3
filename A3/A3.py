from typing import NewType, final
import matplotlib.pyplot as plt
from operator import itemgetter, pos
import sys
import copy
import numpy as np

#let us denote the 6 actions as N-North , S-south, W-west, E-east, PU-Pickup, PD-put down. And let us assume that all the 6 actions are available for each state.
#let us suppose that 0 (in [Taxi_Pos, Pass_Pos,0]) denotes that passenger is not in the taxi.
#let us suppose that 1 (in [Taxi_Pos, Pass_Pos,0]) denotes that passenger is in the taxi.
#[0,0] , [0,3] , [4,0] and [4,4] are the depots.
#The State [45,45] shows that the passenger is in taxi.
#let An end be [[45,45],[45,45],[45,45]] is the end state.

#let us denote discount factor to be Disc_fac
Depos = [[0,0],[0,3],[4,0],[4,4]]

Space = []
for i in range(25):
    row = i//5
    column = i%5
    for x in range(5):                                          #End-20 rem-1  --> 1(20+0)
        for y in range(5):                                  #[0,0],[45,45],[0,0]-->PD. -->END---[45,45],[45,45],[45,45] 2601elem.
            for z in Depos:
                Space.append([[row,column],[x,y],z])
    for dep in Depos:
        Space.append([[row,column],[45,45],dep])

Space.append([[45,45],[45,45],[45,45]])


example_Space = []
for i in range(25):
    row = i//5
    column = i%5
    if [row,column] not in Depos:
        for x in Depos:
            for y in Depos:
                if y!=x:
                    example_Space.append([[row,column],x,y])
#print(len(example_Space))

#print(Space[100])
#print(Space[0])
#print(Space[137])
#print(Space[241])
    
#print(len(Space))

def Graph_plot(Graph_points):
    x = [i[0] for i in Graph_points]
    y = [i[1] for i in Graph_points]
    plt.plot(x,y)
    plt.xlabel("Iteration")
    plt.ylabel("Max-Norm Value")
    plt.title("Graph between Iteration Vs Max-Norm Distance")
    plt.show()  

def max_norm(Current_val, Prev_val):
    s= -2000
    for i in range(2601):
        s = max(s,Current_val[i]-Prev_val[i])
    return s

def next_pos(State, action):
    if action == 'N':
        return [[State[0][0]+1,State[0][1]],State[1],State[2]]
    elif action == 'S':
        return [[State[0][0]-1, State[0][1]],State[1], State[2]]
    elif action == 'W':
        return [[State[0][0], State[0][1]-1],State[1],State[2]]
    elif action == 'E':
        return [[State[0][0] , State[0][1]+1], State[1], State[2]]



def valid_actions(state):
    p= state[0]
    if p == [0,0]:
        return ['N','PU','PD']
    elif p == [1,0]:
        return ['N','S','PU','PD']
    elif p ==[4,3]:
        return ['S','W','E','PU','PD']
    elif p == [0,1] or p == [0,3]:
        return ['N', 'E','PU','PD']
    elif p == [0,2] or p == [0,4]:
        return ['N','W','PU','PD']
    elif p == [4,0] or p == [4,2]:
        return ['S','E','PU','PD']
    elif p == [4,1] or p == [4,4]:
        return ['S','W','PU','PD']
    elif p == [1,1] or p == [1,3] or p == [2,0] or p == [3,0] or p == [3,2]:
        return ['N','S','E','PU','PD']
    elif p == [1,2] or p == [2,4] or p == [3,1] or p == [3,4] or p == [1,4]:
        return ['N','S','W','PU','PD']
    elif p == [2,1] or p == [2,2] or p == [2,3] or p == [2,4] or p == [3,3]:
        return ['N','S','W','E','PU','PD']

    return "NO-VALID-ACTIONS"


def next_state(State,action):
    act_valid = valid_actions(State)
    empt =[]

    for actions in ['N','S','W','E']:
        if actions in act_valid:
            nxt = next_pos(State,actions)
            if actions == action:
                empt.append([nxt,0.85])
            else:
                empt.append([nxt,0.05])             #N - [[new,0.85],[state,0.05],[],[]] new -[[],[],[]]
        else:
            if actions == action:
                empt.append([State,0.85])
            else:
                empt.append([State,0.05])

    return empt

def find_opt_policy(Q_val):
    policy = [0]*2601
    actions = ['N','S','E','W','PU','PD']
    for i in range(2600):
        state = Space[i]
        policy[i] = actions[Q_val[i].index(max(Q_val[i]))]
    return policy
                


    return policy
        
def Policy_Update(policy, State_value,Dist_fact):
    New_policy = [0]*2601
    for i in range(2600):
        state = Space[i]
        list_req = Value(state,State_value,Dist_fact)
        New_policy[i] = list_req[0][1]

    return New_policy



def Value(State, val_st,Dist_fact):
    actions = ['N','S','W','E','PU','PD']
    val_list = []
    for act in actions:
        #print(val_list)
        if act not in ['PU','PD']:
            #print(State,act)
            req = next_state(State, act)

            val = 0
            for i in req:
                #print(val_st[Space.index(i[0])])
                val += i[1]*(-1 + Dist_fact*val_st[Space.index(i[0])])
            val_list.append([val,act])

        elif act == 'PU':
            va = 0
            #state[1] --->  3 cases.
            if State[1] == [45,45]:
                va = -1 + Dist_fact*val_st[Space.index(State)]
            elif State[1] == State[0]:
                va = -1 + Dist_fact*val_st[Space.index([State[0], [45,45], State[2]])]
            elif State[1] != State[0]:
                va = -10 + Dist_fact*val_st[Space.index(State)]
            
            val_list.append([va,act])
        elif act == 'PD':
            #state[1] --> 3 cases.
            v = 0
            if State[1] == [45,45]:
                if State[0] == State[2]:
                    v = 20 + Dist_fact*20
                else:
                    v = -1 + Dist_fact*val_st[Space.index([State[0],State[0],State[2]])]
            elif State[1] == State[0]:
                v = -1 + Dist_fact*val_st[Space.index(State)]
            elif State[1] != State[0]:
                v = -10 + Dist_fact*val_st[Space.index(State)]

            val_list.append([v,act])


    Final_list = sorted(val_list,key = itemgetter(0),reverse=True)
    #print(Final_list[0])
    return Final_list

def Value_iteration(Pass_Pos, Dest_Pos, epsilon,Disc_Fact):
    
    if Pass_Pos not in Depos or Dest_Pos not in Depos:
        return "IN-CORRECT_INPUT"

    Value_state = [-1 for i in range(2601)]
    Prev_valu = [0 for i in range(2601)]
    Value_state[2600] = 20
    Prev_valu[2600] =  0

    iteration = 1
    graph_points = [[1,20]]
    #counter = max_norm(Value_state,Prev_valu)
    while max_norm(Value_state,Prev_valu) > epsilon:
        print(iteration)
        graph_points.append([iteration,max_norm(Value_state,Prev_valu)])
        #print(counter)
        iteration+=1
        policy =[0 for x in range(2601)]
        
        NewVal_St =[0 for x in range(2601)]
        NewVal_St[2600] = 20
        for states in range(2600):
            pair = Value(Space[states],Value_state,Disc_Fact)
                #print(pair)
            NewVal_St[states] = pair[0][0]
            policy[states] = pair[0][1]
        #print(NewVal_St)
        #print(Value_state)
        #counter = max_norm(NewVal_St,Value_state)
        Prev_valu = copy.deepcopy(Value_state)
        Value_state = copy.deepcopy(NewVal_St)
        
    print(iteration)
    #print(graph_points)
    for i in range(2600):
        print(str(Space[i]) + " : " + str(policy[i]))
    #return policy
    return graph_points

#2.a
#print(Value_iteration([0,3],[4,4],0.1,0.9))         #iterations --> 28.

#2.b -1
#print(Value_iteration([0,3],[4,4],0.1,0.01))       #iterations = 4
#graph_points = [[1, 20], [1, 20], [2, 21.2], [3, 0.21199999999999997]]
#Graph_plot(Value_iteration([0,3],[4,4],0.1,0.99))

#2.b -2
#print(Value_iteration([0,3],[4,4],0.1,0.1))        #iterations = 5
#graph_points = [[1, 20], [1, 20], [2, 23.0], [3, 2.3000000000000003], [4, 0.19535000000000013]]                                      
#Graph_plot(graph_points)

#2.b -3
#print(Value_iteration([0,3],[4,4],0.1,0.5))        #iterations = 10
#graph_points = [[1, 20], [1, 20], [2, 31.0], [3, 15.5], [4, 6.5687500000000005], [5, 2.9496875000000005], [6, 1.3200507812500002], [7, 0.5924603515625], [8, 0.26195869873046873], [9, 0.11239007348632812]]
#Graph_plot(graph_points)                                                  

#2.b-4
#print(Value_iteration([0,3],[4,4],0.1,0.8))        #iterations = 21
#graph_points = [[1, 20], [1, 20], [2, 37.0], [3, 29.6], [4, 20.051200000000005], [5, 14.395904000000003], [6, 10.299381760000003], [7, 7.389340467200002], [8, 5.221496406016001], [9, 3.578394635141121], [10, 2.5546975494012933], [11, 1.8286448173289296], [12, 1.288334491431447], [13, 0.9067454541278521], [14, 0.6178493053784688], [15, 0.46656597956971213], [16, 0.35891769765558745], [17, 0.268692949397483], [18, 0.20327762964266327], [19, 0.15123105905879797], [20, 0.11415700319488842]]
#Graph_plot(graph_points)                                                

#2.b -5
#print(Value_iteration([0,3],[4,4],0.1,0.99))       #iterations = 35
#graph_points = [[1, 20], [1, 20], [2, 40.8], [3, 40.391999999999996], [4, 33.844323149999994], [5, 30.05923232564999], [6, 26.6023724692411], [7, 23.60856585515372], [8, 20.63288887864654], [9, 17.484302873706103], [10, 15.436542754911404], [11, 13.663521387125177], [12, 11.901223156643946], [13, 10.354215697105593], [14, 8.717040003702692], [15, 8.120000271389332], [16, 7.7739802511090215], [17, 7.168212828613831], [18, 6.7192495559842], [19, 6.182377903757972], [20, 5.722835873447007], [21, 5.287100452816151], [22, 4.414883631619377], [23, 4.256469119686236], [24, 2.9401619822431737], [25, 2.8497202485754323], [26, 1.9593318292118296], [27, 1.6638529543039162], [28, 1.1174629921512818], [29, 0.8646864995236143], [30, 0.5694010224239161], [31, 0.4133766765360747], [32, 0.2676764305596997], [33, 0.1856083494732257], [34, 0.11846281726988295]]
#Graph_plot(graph_points)                                                   


         
def Policy_iteration(Pass_pos, Dest_pos, Dist_Fact):
    if Pass_pos not in Depos or Dest_pos not in Depos:
        return "IN-CORRECT-TEXT"
    State_val = [0]*2601
    State_val[2600] = 20
    Policy = ['N' for i in range(2601)]
    New_val = [-1 for i in range(2601)]
    New_val[2600] = 20

    final_val=[]
    #final_val.append(New_val)

    New_policy = Policy_Update(Policy, New_val,Dist_Fact)
    iteration =1
    #Val_pol = Value_iteration(Pass_pos, Dest_pos, epsilon)
    #while New_policy != Policy:
    while Policy!= New_policy:
        iteration +=1 
        print(iteration)
        #print(New_policy[2599])
        #for i in range(2600):
            #if Policy[i] != New_policy[i]:
                #print(iteration, i, Policy[i],New_policy[i])
        nval = [0 for i in range(2601)]
        nval[2600] = 20

        for i in range(2600):
            state = Space[i]
            st_val = Value(state, New_val,Dist_Fact)
            for pair in st_val:
                if pair[1] == New_policy[i]:
                    nval[i] = pair[0]

        final_val.append(nval)
        New_val = copy.deepcopy(nval)
        Policy = copy.deepcopy(New_policy)
        New_policy = Policy_Update(Policy,nval,Dist_Fact)

    Graph_points =[]
    optimal_val = final_val[-1]
    for i in range(len(final_val)):
        x = max_norm(optimal_val,final_val[i])
        Graph_points.append([i+1,x])

    #return New_policy
    return Graph_points

#Graph_plot(Policy_iteration([0,3],[4,4],0.1,0.99))

def a_3b(Pass_pos, Dest_pos):
    Points1 = Policy_iteration(Pass_pos, Dest_pos, 0.01)
    Points2 = Policy_iteration(Pass_pos, Dest_pos, 0.1)
    Points3 = Policy_iteration(Pass_pos, Dest_pos, 0.5)
    Points4 = Policy_iteration(Pass_pos, Dest_pos, 0.8)
    Points5 = Policy_iteration(Pass_pos, Dest_pos, 0.99)

    len1 = len(Points1)
    x1 = [Points1[i][0] for i in range(len1)]
    y1 = [Points1[i][1] for i in range(len1)]

    len2 = len(Points2)
    x2 = [Points2[i][0] for i in range(len2)]
    y2 = [Points2[i][1] for i in range(len2)]

    len3 = len(Points3)
    x3 = [Points3[i][0] for i in range(len3)]
    y3 = [Points3[i][1] for i in range(len3)]

    
    len4 = len(Points4)
    x4 = [Points4[i][0] for i in range(len4)]
    y4 = [Points4[i][1] for i in range(len4)]

    len5 = len(Points5)
    x5 = [Points5[i][0] for i in range(len5)]
    y5 = [Points5[i][1] for i in range(len5)]



    #print(y1[-1])              #Value = 9.36117989816859
    #print(y2[-1])              #value = 8.900287722684137
    #print(y3[-1])              #value = 9.143250124034626
    #print(y4[-1])              #value = 9.238662226530009
    plt.plot(x1,y1)
    plt.plot(x2,y2)
    plt.plot(x3,y3)
    plt.plot(x4,y4)
    plt.plot(x5,y5)

    plt.xlabel("Iteration_Number")
    plt.ylabel("Policy_Loss")
    plt.title("Graph between Iteration Vs Policy_Loss for 4 Disc_fact values")
    plt.legend(["Disc_fact = 0.01","Disc_fact = 0.1", "Disc_fact = 0.5", "Disc_fact = 0.8","Disc_fact = 0.99"], loc = "upper right")

    plt.show()

#a_3b([0,3],[4,4])



def simulator(State, action):
    actions = ['N','S','W','E']
    req_act = None
    if action in actions:
        prob = [0.05 for i in range(4)]
        prob[actions.index(action)] = 0.85
        req_act = np.random.choice(actions,size= None, replace = True, p = prob)
        act_valid = valid_actions(State)
        if req_act in act_valid:
            new_state = next_pos(State,req_act)
        else:
            new_state = State
        return [new_state,-1]

    elif action == 'PU':
        result_l = None
        if State[1] == [45,45]:
            result_l = [State,-1]
        elif State[1] == State[0]:
            result_l = [[State[0],[45,45],State[2]],-1]
        elif State[1] != State[0]:
            result_l = [State,-10]
        return result_l

    elif action == 'PD':
        resul = None
        if State[1] == [45,45]:
            if State[0] == State[2]:
                resul = [[[45,45],[45,45],[45,45]], 20]
            else:
                resul = [[State[0],State[0],State[2]],-1]
        elif State[0] == State[1]:
            resul = [State,-1]
        elif State[0] != State[1]:
            resul = [State,-10]
        return resul

#print(simulator([[4,4],[45,45],[4,4]],'S'))

def q_learning_n(epsilon,alpha,episodes,Disc_fact,Dest_pos):
    Q_Value = [[0,0,0,0,0,0] for i in range(2601)]
    Q_Value[2600] =[20,20,20,20,20,20]
    Graph_points = []
    #Q_Value.append([0,0,0,0,0,20])
    actions = ['N','S','E','W','PU','PD']
    empt = []
    for i in range(2601):
        if Space[i][2] == Dest_pos:
            empt.append(i)
    #print(empt)

    State_space = []
    for i in empt:
        if Space[i][1] != [45,45] and Space[i][1] != Dest_pos:
            State_space.append(i)

    for episode in range(episodes):
        print(episode)
        Random_state = Space[np.random.choice(empt)]
        iteration = 0
        while Random_state != [[45,45],[45,45],[45,45]] and iteration<500:
            X_co = Space.index(Random_state)
            iteration += 1
            opt_val = max(Q_Value[X_co])
            opt_act = actions[Q_Value[X_co].index(opt_val)]
            random_act = np.random.choice(actions)
            act_set = [opt_act,random_act]
            Final_act = np.random.choice(a=act_set,p=[1-epsilon,epsilon])
            
            Y_co = actions.index(Final_act)
            New_state = simulator(Random_state,Final_act)
            req_val = max(Q_Value[Space.index(New_state[0])])            
            Q_Value[X_co][Y_co] = (1-alpha)*(Q_Value[X_co][Y_co]) + alpha*(New_state[1] + (Disc_fact*req_val))

            Random_state = New_state[0]

        if episode%100 == 0:
            Opt_Policy = find_opt_policy(Q_Value)

            expected_reward = 0
            for examples in range(200):
                Random_example = Space[np.random.choice(State_space)]
                iter = 0
                Disc_reward = 0
                while Random_example != [[45,45],[45,45],[45,45]] and iter<500:
                    InDex = Space.index(Random_example)
                    #print(Random_example)
                    if Opt_Policy[InDex] in ['N','W','E','S']:
                        #print(Opt_Policy[InDex])
                        Next_random_example = [i[0] for i in next_state(Random_example,Opt_Policy[InDex]) if i[1] == 0.85]
                        Next_random_example.append(-1)
                        #print(Next_random_example)

                    else:
                        Next_random_example = simulator(Random_example,Opt_Policy[InDex])
                    
                    Disc_reward += (Disc_fact**iter)*Next_random_example[1]
                    iter+=1
                    Random_example = Next_random_example[0]
                expected_reward += Disc_reward

            new_exp = expected_reward/200
            Graph_points.append([episode+1,new_exp])

    #print(Graph_points)
    s = find_opt_policy(Q_Value)
    return s
    #for i in empt:
        #print(str(Space[i]) + ":" + str(s[i]) +" " + str(Q_Value[i]))
    #return Graph_points

#policy = q_learning_n(0.1,0.25,5000,0.25,[4,4])
#s1 = [[4,1],[0,0],[4,4]]
#iter =0
#Disc_reward =0
#while s1 != [[45,45],[45,45],[45,45]] and iter<500:
    #InDex = Space.index(s1)
                    #print(Random_example)
    #if policy[InDex] in ['N','W','E','S']:
                        #print(Opt_Policy[InDex])
        #Next_random_example = [i[0] for i in next_state(s1,policy[InDex]) if i[1] == 0.85]
        #Next_random_example.append(-1)
                        #print(Next_random_example)

    #else:
        #Next_random_example = simulator(s1,policy[InDex])
                    
    #Disc_reward += (0.99**iter)*Next_random_example[1]
    #iter+=1
    #s1 = Next_random_example[0]
#print(Disc_reward)

def q_learning_exp(epsilon,alpha,episodes,Disc_fact,Dest_pos):
    Q_Value = [[0,0,0,0,0,0] for i in range(2601)]
    Q_Value[2600] = [20,20,20,20,20,20]
    actions = ['N','S','E','W','PU','PD']
    empt = []
    Graph_points = []
    for i in range(2601):
        if Space[i][2] == Dest_pos:
            empt.append(i)
    State_space = []
    for i in empt:
        if Space[i][1] != [45,45] and Space[i][1] != Dest_pos:
            State_space.append(i)

    for episode in range(episodes):
        mod_eps = copy.deepcopy(epsilon)/(episode+1)
        Random_state = Space[np.random.choice(empt)]
        print(episode)

        iteration = 0

        while Random_state != [[45,45],[45,45],[45,45]] and iteration<500:
            X_co = Space.index(Random_state)
            #print(episode,iteration)
            iteration += 1
            
            opt_val = max(Q_Value[X_co])
            opt_act = actions[Q_Value[X_co].index(opt_val)]
            random_act = np.random.choice(actions)
            act_set = [opt_act,random_act]
            Final_act = np.random.choice(a=act_set,p=[1-mod_eps,mod_eps])
            
            Y_co = actions.index(Final_act)
            New_state = simulator(Random_state,Final_act)
            #print(Random_state,Final_act)
            req_val = max(Q_Value[Space.index(New_state[0])])
            #for act in range(6):
                #req_val = max(req_val, Q_Value[Space.index(New_state[0])][act])
            Q_Value[X_co][Y_co] = (1-alpha)*(Q_Value[X_co][Y_co]) + alpha*(New_state[1] + (Disc_fact*req_val))
            Random_state = New_state[0]

        if episode%100 == 0:
            
            Opt_Policy = find_opt_policy(Q_Value)

            expected_reward = 0
            for examples in range(200):
                Random_example = Space[np.random.choice(State_space)]
                iter = 0
                Disc_reward = 0
                while Random_example != [[45,45],[45,45],[45,45]] and iter<500:
                    InDex = Space.index(Random_example)
                    #print(Random_example)
                    if Opt_Policy[InDex] in ['N','W','E','S']:
                        #print(Opt_Policy[InDex])
                        Next_random_example = [i[0] for i in next_state(Random_example,Opt_Policy[InDex]) if i[1] == 0.85]
                        Next_random_example.append(-1)
                        #print(Next_random_example)
                    else:
                        Next_random_example = simulator(Random_example,Opt_Policy[InDex])

                    Disc_reward += (Disc_fact**iter)*Next_random_example[1]
                    iter+=1
                    Random_example = Next_random_example[0]
                expected_reward += Disc_reward

            #new_exp = expected_reward/200
            Graph_points.append([episode+1,expected_reward/200])

    #print(Graph_points)
    #s = find_opt_policy(Q_Value)
    #return s
    #for i in empt:
        #print(str(Space[i]) + ":" + str(s[i]))
    return Graph_points

#print(q_learning_exp(0.1,0.25,5000,0.99,[4,4]))


def Sarsa(epsilon, alpha, episodes, Disc_fact,Dest_pos):
    Q_Value = [[0,0,0,0,0,0] for i in range(2601)]
    actions = ['N','S','E','W','PU','PD']
    empt = []
    Graph_points = []
    for i in range(2601):
        if Space[i][2] == Dest_pos:
            empt.append(i)

    State_space = []
    for i in empt:
        if Space[i][1] != [45,45] and Space[i][1] != Dest_pos:
            State_space.append(i)
    for episode in range(episodes):
        print(episode)
        Random_state = Space[np.random.choice(empt)]
        #print(Random_state)
        iteration = 0
        while Random_state != [[45,45],[45,45],[45,45]] and iteration<500:
            X_co = Space.index(Random_state)
            #print(episode,iteration)
            iteration += 1
            
            opt_val = max(Q_Value[X_co])
            opt_act = actions[Q_Value[X_co].index(opt_val)]
            random_act = np.random.choice(actions)
            act_set = [opt_act,random_act]
            Final_act = np.random.choice(a=act_set,p=[1-epsilon,epsilon])
            
            Y_co = actions.index(Final_act)
            New_state = simulator(Random_state,Final_act)
            usedval = Space.index(New_state[0])
            req_val = max(Q_Value[Space.index(New_state[0])])
            new_opt = actions[Q_Value[usedval].index(req_val)]
            new_rand = np.random.choice(actions)
            new_set = [new_opt,new_rand]
            new_final = np.random.choice(new_set,p=[1-epsilon,epsilon])
            new_f_val = Q_Value[usedval][actions.index(new_final)]
            
            Q_Value[X_co][Y_co] = (1-alpha)*(Q_Value[X_co][Y_co]) + alpha*(New_state[1] + (Disc_fact*new_f_val))
            Random_state = New_state[0]
            
            
        if episode%100 == 0:
            
            Opt_Policy = find_opt_policy(Q_Value)
            expected_reward = 0
            for examples in range(200):
                Random_example = Space[np.random.choice(State_space)]
                iter = 0
                Disc_reward = 0
                while Random_example != [[45,45],[45,45],[45,45]] and iter<500:
                    InDex = Space.index(Random_example)
                    #print(Random_example)
                    if Opt_Policy[InDex] in ['N','W','E','S']:
                        #print(Opt_Policy[InDex])
                        Next_random_example = [i[0] for i in next_state(Random_example,Opt_Policy[InDex]) if i[1] == 0.85]
                        Next_random_example.append(-1)
                        #print(Next_random_example)
                    else:
                        Next_random_example = simulator(Random_example,Opt_Policy[InDex])

                    #New_random_example = simulator(Random_example,Opt_Policy[InDex])
                    Disc_reward += (Disc_fact**iter)*Next_random_example[1]
                    iter+=1
                    Random_example = Next_random_example[0]
                expected_reward += Disc_reward

            new_exp = expected_reward/200
            Graph_points.append([episode+1,new_exp])
    #print(Graph_points)
    #s = find_opt_policy(Q_Value)
    #return s
    #for i in empt:
        #print(str(Space[i]) + ":" + str(s[i]))
    return Graph_points
    

#print(Sarsa(0.1,0.25,5000,0.99,[4,4]))

def Sarsa_exp(epsilon, alpha, episodes, Disc_fact,Dest_pos):
    Q_Value = [[0,0,0,0,0,0] for i in range(2601)]
    actions = ['N','S','E','W','PU','PD']
    empt = []
    Graph_points =[]
    for i in range(2601):
        if Space[i][2] == Dest_pos:
            empt.append(i)

    State_space = []
    for i in empt:
        if Space[i][1] != [45,45] and Space[i][1] != Dest_pos:
            State_space.append(i)
    for episode in range(episodes):
        print(episode)
        mod_eps = copy.deepcopy(epsilon)/(episode+1)
        Random_state = Space[np.random.choice(empt)]
        #print(Random_state)
        iteration = 0
        while Random_state != [[45,45],[45,45],[45,45]] and iteration<500:
            X_co = Space.index(Random_state)
            iteration += 1
            opt_val = max(Q_Value[X_co])
            opt_act = actions[Q_Value[X_co].index(opt_val)]
            random_act = np.random.choice(actions)
            act_set = [opt_act,random_act]
            Final_act = np.random.choice(a=act_set,p=[1-mod_eps,mod_eps])
            
            Y_co = actions.index(Final_act)
            
            New_state = simulator(Random_state,Final_act)
            usedval = Space.index(New_state[0])
            req_val = max(Q_Value[Space.index(New_state[0])])
            new_opt = actions[Q_Value[usedval].index(req_val)]
            new_rand = np.random.choice(actions)
            new_set = [new_opt,new_rand]
            new_final = np.random.choice(new_set,p=[1-mod_eps,mod_eps])
            new_f_val = Q_Value[usedval][actions.index(new_final)]
            Q_Value[X_co][Y_co] = (1-alpha)*(Q_Value[X_co][Y_co]) + alpha*(New_state[1] + (Disc_fact*new_f_val))
            Random_state = New_state[0]

        if episode%100 == 0:
            
            Opt_Policy = find_opt_policy(Q_Value)

            expected_reward = 0
            for examples in range(200):
                Random_example = Space[np.random.choice(State_space)]
                iter = 0
                Disc_reward = 0
                while Random_example != [[45,45],[45,45],[45,45]] and iter<500:
                    InDex = Space.index(Random_example)
                    if Opt_Policy[InDex] in ['N','W','E','S']:
                        Next_random_example = [i[0] for i in next_state(Random_example,Opt_Policy[InDex]) if i[1] == 0.85]
                        Next_random_example.append(-1)
                    else:
                        Next_random_example = simulator(Random_example,Opt_Policy[InDex])

                    Disc_reward += (Disc_fact**iter)*Next_random_example[1]
                    iter+=1
                    Random_example = Next_random_example[0]
                expected_reward += Disc_reward

            new_exp = expected_reward/200
            Graph_points.append([episode+1,new_exp])

    #print(Graph_points)
    #s = find_opt_policy(Q_Value)
    #return s
    #for i in empt:
        #print(str(Space[i]) + ":" + str(s[i]))
    return Graph_points

#print(Sarsa_exp(0.1,0.25,5000,0.99,[4,4]))

def b_2(epsilon, alpha,episodes,Disc_fact,Dest_pos):
    Points1 = q_learning_n(epsilon,alpha,episodes,Disc_fact,Dest_pos)
    Points2 = q_learning_exp(epsilon,alpha,episodes,Disc_fact,Dest_pos)
    points3 = Sarsa(epsilon,alpha,episodes,Disc_fact,Dest_pos)
    points4 = Sarsa_exp(epsilon,alpha,episodes,Disc_fact,Dest_pos)

    len1 = len(Points1)
    x1 = [Points1[i][0] for i in range(len1)]
    y1 = [Points1[i][1] for i in range(len1)]

    len2 = len(Points2)
    x2 = [Points2[i][0] for i in range(len2)]
    y2 = [Points2[i][1] for i in range(len2)]

    len3 = len(points3)
    x3 = [points3[i][0] for i in range(len3)]
    y3 = [points3[i][1] for i in range(len3)]

    len4 = len(points4)
    x4 = [points4[i][0] for i in range(len4)]
    y4 = [points4[i][1] for i in range(len4)]

    #print(y1[-1])              #Value = 9.36117989816859
    #print(y2[-1])              #value = 8.900287722684137
    #print(y3[-1])              #value = 9.143250124034626
    #print(y4[-1])              #value = 9.238662226530009
    plt.plot(x1,y1)
    plt.plot(x2,y2)
    plt.plot(x3,y3)
    plt.plot(x4,y4)

    plt.xlabel("Episode_Number")
    plt.ylabel("Average_Discounted_reward")
    plt.title("Average Discounted Reward vs Episdoes for the 4 Algorithms")
    plt.legend(["Q-Learning","Q-Learning(Decay)", "Sarsa", "Sarsa(Decay)"], loc = "lower right")

    plt.show()

#b_2(0.1,0.25,5000,0.99,[4,4])

def b_4(episodes, Disc_fact, Dest_pos):
    points1 = q_learning_n(0,0.1,episodes,Disc_fact,Dest_pos)
    points2 = q_learning_n(0.05,0.1,episodes,Disc_fact,Dest_pos)
    points3 = q_learning_n(0.1,0.1,episodes,Disc_fact,Dest_pos)
    points4 = q_learning_n(0.5,0.1,episodes,Disc_fact,Dest_pos)
    points5 = q_learning_n(0.9,0.1,episodes,Disc_fact,Dest_pos)

    len1 = len(points1)
    x1 = [points1[i][0] for i in range(len1)]
    y1 = [points1[i][1] for i in range(len1)]

    len2 = len(points2)
    x2 = [points2[i][0] for i in range(len2)]
    y2 = [points2[i][1] for i in range(len2)]

    len3 = len(points3)
    x3 = [points3[i][0] for i in range(len3)]
    y3 = [points3[i][1] for i in range(len3)]

    len4 = len(points4)
    x4 = [points4[i][0] for i in range(len4)]
    y4 = [points4[i][1] for i in range(len4)]

    len5 = len(points5)
    x5 = [points5[i][0] for i in range(len5)]
    y5 = [points5[i][1] for i in range(len5)]

    plt.plot(x1,y1)
    plt.plot(x2,y2)
    plt.plot(x3,y3)
    plt.plot(x4,y4)
    plt.plot(x5,y5)

    plt.xlabel("Episode_Number")
    plt.ylabel("Average_Discounted_reward")
    plt.title("Average Discounted Reward vs Episdoes for the 5 epsilon values")
    plt.legend(["epsilon=0","epsilon=0.05", "epsilon=0.1", "epsilon=0.5", "epsilon=0.9"], loc = "lower right")

    plt.show()

#b_4(5000,0.99,[4,4])


def b_42(episodes, Disc_fact, Dest_pos):
    points1 = q_learning_n(0.1,0.1,episodes,Disc_fact,Dest_pos)
    points2 = q_learning_n(0.1,0.2,episodes,Disc_fact,Dest_pos)
    points3 = q_learning_n(0.1,0.3,episodes,Disc_fact,Dest_pos)
    points4 = q_learning_n(0.1,0.4,episodes,Disc_fact,Dest_pos)
    points5 = q_learning_n(0.1,0.5,episodes,Disc_fact,Dest_pos)

    len1 = len(points1)
    x1 = [points1[i][0] for i in range(len1)]
    y1 = [points1[i][1] for i in range(len1)]

    len2 = len(points2)
    x2 = [points2[i][0] for i in range(len2)]
    y2 = [points2[i][1] for i in range(len2)]

    len3 = len(points3)
    x3 = [points3[i][0] for i in range(len3)]
    y3 = [points3[i][1] for i in range(len3)]

    len4 = len(points4)
    x4 = [points4[i][0] for i in range(len4)]
    y4 = [points4[i][1] for i in range(len4)]

    len5 = len(points5)
    x5 = [points5[i][0] for i in range(len5)]
    y5 = [points5[i][1] for i in range(len5)]

    plt.plot(x1,y1)
    plt.plot(x2,y2)
    plt.plot(x3,y3)
    plt.plot(x4,y4)
    plt.plot(x5,y5)

    plt.xlabel("Episode_Number")
    plt.ylabel("Average_Discounted_reward")
    plt.title("Average Discounted Reward vs Episdoes for the 5 alpha values")
    plt.legend(["alpha=0.1","alpha=0.2", "alpha=0.3", "alpha=0.4", "alpha=0.5"], loc = "lower right")

    plt.show()


#b_42(5000,0.99,[4,4])

#all the below functiona are for 10x10 matrix.
#here depos are [1,0],[0,4],[0,9],[6,3],[5,6],[9,0],[9,5],[9,8]

def last_next_pos(State, action):
    if action == 'N':
        return [State[0]+1,State[1]]
    elif action == 'S':
        return [State[0]-1, State[1]]
    elif action == 'W':
        return [State[0], State[1]-1]
    elif action == 'E':
        return [State[0] , State[1]+1]


def last_action(state):
    actions = []
    if state[0] in [0,1,2,3,4,5,6,7,8]:
        actions.append('N')
    if state[0] in [1,2,3,4,5,6,7,8,9]:
        actions.append('S')
    if state[1] in [0,1,2,3,4,5,6,7,8] and (state[0] not in [0,1,2,3] or state[1] not in [0,3,7]) and (state[0] not in [4,5,6,7] or state[1]!=5) and (state[0] not in [6,7,8,9] or state[1] not in [2,7]):
        actions.append('E')
    if state[1] in [1,2,3,4,5,6,7,8,9] and (state[0] not in [0,1,2,3] or state[1] not in [1,4,8]) and (state[0] not in [4,5,6,7] or state[1]!=6) and (state[0] not in [6,7,8,9] or state[1] not in [3,8]):
        actions.append('W')

    return actions

def last_sim(state,action):
    val_actions = last_action(state)
    actions = ['N','S','E','W']
    empt = []
    for act in actions:
        if act in val_actions:
            empt.append([act,last_next_pos(state,act)])
        else:
            empt.append([act,state])

    if act in ['N','S','E','W']:
        prob =[0.05 for i in range(4)]
        prob[actions.index(action)] = 0.85
        random_act = np.random.choice(actions,p=prob)


#print(last_sim([0,0],'S'))









    

    


   


    



    






