'''    '''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf


#### Reading CoM work compoents gains
# com_gains = pd.read_excel('D:/Foundamentals/Preferred Walking Speed Based On CoM Work Components/CoMWorksGains2.xlsx')
v_ave    = np.linspace(0.8, 1.6, 100)

###
## initial guess
alpha_ini = 0.5 * v_ave ** 0.42

## finding alpha when v_ave = 1.5m/s
def find_index_larger_than(arr, target):
    for i in range(len(arr)):
        if arr[i] > target:
            return i
    # If no element in the array is larger than the target
    return -1

v_target = find_index_larger_than(v_ave, 1.5)
c = 1 / (alpha_ini[v_target] / 0.4)

## Adjusted Alpha
alpha = 0.5 * c * v_ave**0.42


com_gains = pd.read_csv(
    'CoMWorksGains2.csv',
    sep=',',  # Use ';' if your file uses semicolon as a delimiter
    header=0,  # Use the first row as the header
    index_col=0,  # Use the first column as the index
    )

## Sanity check
## Calculating the COM work components for young adult with normal lookahead 
## work = intercept + vision + vel-gain * velo**3
vel_range = np.linspace(0.8, 1.6, 100)

p_ya_w  = com_gains['P'][0]  + com_gains['P'][1]  + com_gains['P'][3]  * vel_range**2
n_ya_w  = com_gains['N'][0]  + com_gains['N'][1]  + com_gains['N'][3]  * vel_range**2
po_ya_w = com_gains['PO'][0] + com_gains['PO'][1] + com_gains['PO'][3] * vel_range**2
co_ya_w = com_gains['CO'][0] + com_gains['CO'][1] + com_gains['CO'][3] * vel_range**2
re_ya_w = com_gains['RE'][0] + com_gains['RE'][1] + com_gains['RE'][3] * vel_range**2
pr_ya_w = com_gains['PR'][0] + com_gains['PR'][1] + com_gains['PR'][3] * vel_range**2

## Calculating the COM work components
H = np.array([0, 0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05, 0.055, 0.06])

def sign_changes(arr):
    changes = []
    for i in range(1, len(arr)):
        if (arr[i - 1] >= 0 and arr[i] < 0) or (arr[i - 1] < 0 and arr[i] >= 0):
            changes.append(i - 1)
    return changes

## Sanity check:
index_mid = sign_changes((p_ya_w - po_ya_w) + (n_ya_w - co_ya_w))
print('Preferred walking speed (mid-flight method):                          ', vel_range[index_mid])
index_pre = sign_changes((po_ya_w - (p_ya_w + n_ya_w)) + co_ya_w)
print('Preferred walking speed (pre-emptive method):                         ', vel_range[index_pre])

## Sanity check
# plt.plot(vel_range, po_ya_w, 'r')
# plt.plot(vel_range, -co_ya_w, 'b')
# plt.plot(vel_range,  po_ya_w - (p_ya_w + n_ya_w), color = 'gray')

def pref_vel (age, vision, h):
    
    ## Positive
    p = (com_gains['P'][0] + com_gains['P'][1] * vision + com_gains['P'][2] * age +
      com_gains['P'][3] * vel_range**2 + com_gains['P'][4] * vel_range**2 * age +
      com_gains['P'][5] * h**2 + com_gains['P'][6] * h**2 * age)
    
    ## Negative
    n = (com_gains['N'][0] + com_gains['N'][1] * vision + com_gains['N'][2] * age +
      com_gains['N'][3] * vel_range**2 + com_gains['N'][4] * vel_range**2 * age +
      com_gains['N'][5] * h**2 + com_gains['N'][6] * h**2 * age)
    
    ## Push-off
    po = (com_gains['PO'][0] + com_gains['PO'][1] * vision + com_gains['PO'][2] * age +
      com_gains['PO'][3] * vel_range**2 + com_gains['PO'][4] * vel_range**2 * age +
      com_gains['PO'][5] * h**2 + com_gains['PO'][6] * h**2 * age)
    
    ## Collision
    co = (com_gains['CO'][0] + com_gains['CO'][1] * vision + com_gains['CO'][2] * age +
      com_gains['CO'][3] * vel_range**2 + com_gains['CO'][4] * vel_range**2 * age +
      com_gains['CO'][5] * h**2 + com_gains['CO'][6] * h**2 * age)
    
    ##
    re = p - po
    pr = n - co

    ## Sanity check
    # plt.plot(vel_range, re, 'r', linestyle = '--')
    # plt.plot(vel_range, -pr, 'b', linestyle = '--')

    diff = re + pr

    index = sign_changes(diff)

    vel_preferred = vel_range[index]
    
    return vel_preferred

##
ya_w, ya_nw, oa_w, oa_nw = np.array([]), np.array([]), np.array([]), np.array([])

for h in H:
    #
    ya_w  = np.append(ya_w,  pref_vel(0, 1, h))   
    ya_nw = np.append(ya_nw, pref_vel(0, 0, h))
    
    oa_w  = np.append(oa_w,  pref_vel(1, 1, h))
    oa_nw = np.append(oa_nw, pref_vel(1, 0, h))
    
## Adding the terrain amplitude range
YA_w = np.column_stack((H[: len(ya_w)], ya_w));    
YA_nw = np.column_stack((H[: len(ya_nw)], ya_nw)); 

OA_w = np.column_stack((H[: len(oa_w)], oa_w));    
OA_nw = np.column_stack((H[: len(oa_nw)], oa_nw)); 

ya_w = np.column_stack((H[: len(ya_w)]**2, ya_w));    ya_w_df = pd.DataFrame(ya_w, columns = ['dele', 'vel'])
ya_nw = np.column_stack((H[: len(ya_nw)]**2, ya_nw)); ya_nw_df = pd.DataFrame(ya_nw, columns = ['dele', 'vel'])

oa_w = np.column_stack((H[: len(oa_w)]**2, oa_w));    oa_w_df = pd.DataFrame(oa_w, columns = ['dele', 'vel']) 
oa_nw = np.column_stack((H[: len(oa_nw)]**2, oa_nw)); oa_nw_df = pd.DataFrame(oa_nw, columns = ['dele', 'vel'])

h_range = np.linspace(0, H[len(ya_w)], 100); H_range = h_range**2
h_range_df = pd.DataFrame(H_range, columns = ['dele'])

## fitting a second degree polynomial
estimate_ya_w  = smf.ols('vel ~ dele', data = ya_w_df).fit();  ya_w_pred  = estimate_ya_w.predict(h_range_df)
estimate_ya_nw = smf.ols('vel ~ dele', data = ya_nw_df).fit(); ya_nw_pred = estimate_ya_nw.predict(h_range_df)

estimate_oa_w  = smf.ols('vel ~ dele', data = oa_w_df).fit();  oa_w_pred  = estimate_oa_w.predict(h_range_df)
estimate_oa_nw = smf.ols('vel ~ dele', data = oa_nw_df).fit(); oa_nw_pred = estimate_oa_nw.predict(h_range_df)


## adjustment for the random treadmill calibration bias
bias = (p_ya_w + n_ya_w)

## plotting
plt.figure(1, figsize=(14, 6))

plt.subplot(1,2,1)
plt.plot(vel_range, p_ya_w - bias, linewidth = 2.5, color = 'red', label = 'Total positive work')
plt.plot(vel_range, -n_ya_w, linewidth = 2.5, color = 'blue', label = 'Total negative work', linestyle = '--')


plt.plot(vel_range,  po_ya_w - bias, linewidth = 2.5, color = 'darkred',  label = 'Push-off')
plt.plot(vel_range, -co_ya_w, linewidth = 2.5, color = 'darkblue', label = 'Collision')

plt.plot(vel_range,  p_ya_w - po_ya_w, linewidth = 2.5, color = 'darkorange', label = 'Rebound')
plt.plot(vel_range, -n_ya_w + co_ya_w, linewidth = 2.5, color = 'darkgreen',  label = 'Preload')

plt.legend(frameon=False, fontsize = 12.5)
plt.xticks(fontsize = 12.5); plt.xlabel('Walking Speed (m/s)', fontsize = 12.5)
plt.yticks(fontsize = 12.5); plt.ylabel('Work (J/kg)', fontsize = 12.5)
plt.title('A. Estimating the preferred walking speed by COM work components', fontsize = 12.5, loc = 'left')

##
plt.subplot(1,2,2)
plt.scatter(YA_w[:,0], YA_w[:,1], s = 35, color = 'darkred', label = 'YA - Normal lookahead')
plt.plot(h_range, np.array(ya_w_pred) , color = 'darkred', linewidth = 3)

plt.scatter(YA_nw[:,0], YA_nw[:,1], s = 35, color = 'darkorange', label = 'YA - Restricted lookahead')
plt.plot(h_range, np.array(ya_nw_pred) , color = 'darkorange', linewidth = 3)

plt.scatter(OA_w[:,0], OA_w[:,1], s = 35, color = 'darkgreen', label = 'OA - Normal lookahead')
plt.plot(h_range, np.array(oa_w_pred) , color = 'darkgreen', linewidth = 3)

plt.scatter(OA_nw[:,0], OA_nw[:,1], s = 35, color = 'darkblue', label = 'OA - Restricted lookahead')
plt.plot(h_range, np.array(oa_nw_pred) , color = 'darkblue', linewidth = 3)

plt.legend(frameon=False, fontsize = 12.5)
plt.xticks(fontsize = 12.5); plt.xlabel('Terrain amplitude (m)', fontsize = 12.5)
plt.yticks(fontsize = 12.5); plt.ylabel('Preferred walking speed (m/s)', fontsize = 12.5)
plt.title('B. Estimated preferred walking speed trajectories', fontsize = 12.5, loc = 'left')

plt.tight_layout()