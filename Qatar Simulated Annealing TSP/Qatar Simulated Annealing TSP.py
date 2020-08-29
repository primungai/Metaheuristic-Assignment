
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import random as rd
import datetime

# Upload qatar Travelling Sales Man Problem city coordinates data
qatar_data = pd.read_table('Qatar_TSP.txt', header = None, sep = '\s+')
qatar_data.columns = ['city', 'x', 'y']



# Calculate the distance between the cities using Euclidian Distance
# Place these distances in a dataframe called distance_df
for i in qatar_data["city"]:
    for j in qatar_data["city"]:
        row = qatar_data[qatar_data["city"] == j][["x", "y"]]
        x_coord = row["x"].tolist()[0]
        y_coord = row["y"].tolist()[0]
        qatar_data.loc[qatar_data['city'] == i, j] = ((qatar_data["x"] - x_coord)**2 + (qatar_data["y"] - y_coord)**2)**0.5

df = qatar_data.drop(["x", "y"], axis=1)
distance_df = df.set_index('city')

# Cost matrix associated to travelling to every city
cost_df = distance_df.round(0)

# Initialize a random route to be followed by the travelling salesman
X0 = (np.array(qatar_data["city"]))
rd.shuffle(X0)

# Make a dataframe of the initial solution
# Arrange the cities according to X0 using reindex
New_Dist_DF = distance_df.reindex(columns=X0, index=X0)
New_Dist_Arr = np.array(New_Dist_DF)

# Make a starting dataframe of the objective function using the euclidian distances calculated
# between cities and the cost of travelling between cities
Objfun1_start = pd.DataFrame(New_Dist_Arr*cost_df)
Objfun1_start_Arr = np.array(Objfun1_start)


# sum rows in the array horizontally and then sum the rows in the column containing
# the sum of the rows
sum_start = sum(sum(Objfun1_start_Arr))



T0 = 1500 # Set the initial temperature
M = 250 # Set the number of temperature locations
N = 20 # Set the number of moves at each Temperature
alpha = 0.9 # Set the move operator

# Register the Temperature anc cost that will be plotted at the end of the model run
Temp = []
Min_Cost = []

begin_time = datetime.datetime.now()# start time (timing purposes)



for i in range(M):
    for j in range(N):
        # choose a random number between 0 and the length of the array X0
        # swap the cities to be visited using these random numbers
        ran_1 = np.random.randint(0,len(X0)) 
        ran_2 = np.random.randint(0,len(X0))
        
        while ran_1==ran_2:
            ran_2 = np.random.randint(0,len(X0))
        
        xt = []
    
        # place the 2 cities to be swapped in a temporary object A1 and A2
        A1 = X0[ran_1]
        A2 = X0[ran_2]

        # Make a new list of the new set of cities to be visited
        # swap cities using #append
        w = 0
        for i in X0:
            if X0[w]==A1:
                xt = np.append(xt,A2)
            elif X0[w]==A2:
                xt = np.append(xt,A1)
            else:
                xt=np.append(xt,X0[w]) # if the current city is not one of the cities to be swapped
                                       # append the current city to the xt array
            w = w+1 # move to the next letter in the array
        

        # Arrange the dataframe new_dis_df_init according to the list X0
        new_dis_df_init = distance_df.reindex(columns=X0, index=X0)
        new_dis_init_arr = np.array(new_dis_df_init)

        # Arrange the dataframe new_dis_df_new according to the list xt
        new_dis_df_new = distance_df.reindex(columns=xt, index=xt)
        new_dis_new_arr = np.array(new_dis_df_new)
        
        
        # Make a dataframe of the current solution X0
        objfun_init = pd.DataFrame(new_dis_init_arr*cost_df)
        objfun_init_arr = np.array(objfun_init)
        
        # Make a dataframe of the new solution xt
        objfun_new = pd.DataFrame(new_dis_new_arr*cost_df)
        objfun_new_arr = np.array(objfun_new)
        
        # After swapping cities visited, sum the "sum of rows"
        # sum rows in the array horizontally and then sum vertically the rows in the column containing
        # the sum of the rows
        sum_init = sum(sum(objfun_init_arr))
        sum_new = sum(sum(objfun_new_arr))
        
        # create a random number 
        # calculate the simulated annealing formula
        rand1 = np.random.rand()
        form = 1/(np.exp(sum_new-sum_init)/T0)
        
        # if the new sum of the "sum of rows" is less than or equal to the initial sum of the "sum of rows"
        # Then update the initial array of cities to be exactly the same as the new array of cities 
        # to be visited
        if sum_new<=sum_init:
            X0=xt
            # Else if the random value created is less than or equal to the formula
            # Then update the initial array of cities to be exactly the same as the new array of cities 
            # to be visited
        elif rand1<=form:
            X0=xt
            # Else the initial array of cities to be visited stays the same
        else:
            X0=X0
            
    # Add results to the objects to be used for plotting the final graph    
    Temp.append(T0)
    Min_Cost.append(sum_init)
    
    # Reduce the temperature
    T0 = alpha*T0
    
print(datetime.datetime.now() - begin_time) # end time (timing purposes)

print("Final Solution:",X0)
print("Minimized Cost:",sum_init)
        

plt.plot(Temp,Min_Cost)
plt.title("Cost vs. Temp.", fontsize=20,fontweight='bold')
plt.xlabel("Temp.", fontsize=18,fontweight='bold')
plt.ylabel("Cost", fontsize=18,fontweight='bold')
plt.xlim(1500,0)


plt.xticks(np.arange(min(Temp),max(Temp),100),fontweight='bold')
plt.yticks(fontweight='bold')



